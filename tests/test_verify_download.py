"""The download check (ops/gcp/verify_download.py): every object present with its size and checksums, nothing extra."""
import base64
import hashlib
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("verify_download",
                                              Path(__file__).parents[1] / "ops/gcp/verify_download.py")
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)


def b64md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def fake_sums(p: Path):            # stands in for CRC32C: the test only needs a content-dependent value
    data = p.read_bytes()
    return "crc:" + b64md5(data), b64md5(data)


def setup(tmp_path, files):
    for name, data in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return tmp_path


def obj(name, data, md5=True):
    return (name, len(data), "crc:" + b64md5(data), b64md5(data) if md5 else None)


def test_a_complete_matching_download_passes(tmp_path):
    f = {"pre/a/x.bin": b"abc", "pre/y.bin": b"hello"}
    r = vd.compare([obj(n, d) for n, d in f.items()], setup(tmp_path, f), ["pre"], fake_sums)
    assert r["ok"] and r["objects"] == 2 and r["bytes"] == 8 and r["md5_checked"] == 2


def test_composite_objects_without_md5_are_checked_by_crc32c_only(tmp_path):
    f = {"pre/big.tar": b"tarball"}
    r = vd.compare([obj("pre/big.tar", b"tarball", md5=False)], setup(tmp_path, f), ["pre"], fake_sums)
    assert r["ok"] and r["md5_checked"] == 0


def test_missing_short_corrupt_and_extra_files_fail(tmp_path):
    local = {"pre/ok": b"same", "pre/short": b"abc", "pre/corrupt": b"xbcd", "pre/stray": b"?"}
    bucket = [obj("pre/ok", b"same"), obj("pre/short", b"abcd"), obj("pre/corrupt", b"abcd"), obj("pre/gone", b"z")]
    r = vd.compare(bucket, setup(tmp_path, local), ["pre"], fake_sums)
    assert not r["ok"]
    assert (r["missing"], r["size_mismatch"], r["crc32c_mismatch"], r["md5_mismatch"], r["extra"]) == (
        ["pre/gone"], ["pre/short"], ["pre/corrupt"], ["pre/corrupt"], ["pre/stray"])


def test_an_empty_listing_is_never_ok(tmp_path):
    assert vd.compare([], tmp_path, ["pre"], fake_sums)["ok"] is False


def test_real_crc32c_matches_cloud_storage_encoding(tmp_path):
    google_crc32c = __import__("pytest").importorskip("google_crc32c")
    p = tmp_path / "f"
    p.write_bytes(b"123456789")
    crc, md5 = vd.file_sums(p, google_crc32c.Checksum)
    assert crc == base64.b64encode(bytes.fromhex("e3069283")).decode()     # CRC32C check value of "123456789"
    assert md5 == b64md5(b"123456789")
