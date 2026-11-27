"""The end-of-Phase-1 download (ops/gcp/sync_download.py): fetch only what is missing or different, skip excluded
prefixes, write each file whole (temporary file, then rename), and report failures instead of hiding them."""
import importlib.util
import re
from pathlib import Path

spec = importlib.util.spec_from_file_location("sync_download", Path(__file__).parents[1] / "ops/gcp/sync_download.py")
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)


def crc(p: Path) -> str:                    # stands in for CRC32C: the test only needs a content-dependent value
    return "crc:" + p.read_bytes().hex()


def put(root: Path, name: str, data: bytes):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


class Blob:
    def __init__(self, name, data, fail_times=0):
        self.name, self.data, self.size, self.fail_times, self.calls = name, data, len(data), fail_times, 0

    def download_to_filename(self, path, checksum=None):
        assert checksum == "crc32c"
        self.calls += 1
        Path(path).write_bytes(self.data[: self.size // 2])       # a partial write first, as an interrupted copy
        if self.calls <= self.fail_times:
            raise ConnectionError("reset")
        Path(path).write_bytes(self.data)


def test_only_missing_or_different_objects_are_fetched_and_excluded_prefixes_never(tmp_path):
    put(tmp_path, "v/a/same.bin", b"same")
    put(tmp_path, "v/a/changed.bin", b"old!")
    put(tmp_path, "v/a/short.bin", b"ab")
    objs = [("v/a/same.bin", 4, "crc:" + b"same".hex()), ("v/a/changed.bin", 4, "crc:" + b"new!".hex()),
            ("v/a/short.bin", 3, "crc:" + b"abc".hex()), ("v/a/new.bin", 1, "crc:00"),
            ("v/base/x.bin", 1, "crc:00"), ("v/basement/y.bin", 1, "crc:00"), ("v/dir/", 0, "")]
    got = sd.to_fetch(objs, tmp_path, ["v/base"], crc)
    assert got == ["v/a/changed.bin", "v/a/short.bin", "v/a/new.bin", "v/basement/y.bin"]   # base/ only, not basement/


def test_files_land_whole_retries_recover_and_a_lasting_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(sd.time, "sleep", lambda s: None)
    blobs = {"v/ok.bin": Blob("v/ok.bin", b"0123456789"), "v/flaky.bin": Blob("v/flaky.bin", b"abcdef", fail_times=2),
             "v/dead/x.bin": Blob("v/dead/x.bin", b"zz", fail_times=99)}
    r = sd.fetch_all(blobs, list(blobs), tmp_path, workers=2, tries=3)
    assert (tmp_path / "v/ok.bin").read_bytes() == b"0123456789"
    assert (tmp_path / "v/flaky.bin").read_bytes() == b"abcdef" and blobs["v/flaky.bin"].calls == 3
    assert not (tmp_path / "v/dead/x.bin").exists() and not list(tmp_path.rglob("*.part"))
    assert r["fetched"] == 2 and r["failed"] == 1 and r["failures"][0][0] == "v/dead/x.bin"
    assert "ConnectionError" in r["failures"][0][1] and r["bytes"] == 18


def test_a_second_run_after_more_objects_land_fetches_only_the_new_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(sd.time, "sleep", lambda s: None)
    first = {"v/r/1.json": Blob("v/r/1.json", b"one")}
    sd.fetch_all(first, list(first), tmp_path)
    later = {**first, "v/r/2.json": Blob("v/r/2.json", b"two")}
    objs = [(n, b.size, "crc:" + b.data.hex()) for n, b in later.items()]
    assert sd.to_fetch(objs, tmp_path, [], crc) == ["v/r/2.json"]


def test_the_cli_exclude_regex_is_relative_to_the_source_and_matches_whole_path_parts():
    rx = sd.cli_exclude_regex("vla-router", ["vla-router/images/base", "vla-router/snapshots/base/", "containers/x"])
    assert rx == "^images/base/|^snapshots/base/"
    m = re.compile(rx)
    assert m.search("images/base/shard-1.tar") and m.search("snapshots/base/a/b.npz")
    assert not m.search("images/explore/shard-1.tar") and not m.search("images/basement/x")
    assert sd.cli_exclude_regex("vla-router/records", ["vla-router/images/base"]) is None
