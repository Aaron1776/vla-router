"""The batch-invariance comparison rules (invariance.py, SPEC §17 amendment 33): bit for bit except inside an image on the
CPU, where a difference is a warning if it is within PyTorch's default tolerance and closest to the request's own solo
result, and a failure otherwise."""

import importlib
import warnings

import pytest

np = pytest.importorskip("numpy")

import invariance  # noqa: E402

REF = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
OTHER = np.linspace(1.0, -1.0, 64, dtype=np.float32)          # another request's solo result


def ulps(x, n):
    return (np.asarray(x, dtype=np.float32).view(np.int32) + n).view(np.float32)


@pytest.mark.parametrize("marker, device, expected", [("1", "cpu", True), ("1", "cuda", False), (None, "cpu", False),
                                                      ("1", None, False)])
def test_only_an_image_cpu_run_is_tolerant(monkeypatch, marker, device, expected):
    for name, value in (("VLAR_IN_IMAGE", marker), ("VLAR_TEST_DEVICE", device)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    try:
        assert importlib.reload(invariance).IN_IMAGE_CPU is expected
    finally:
        monkeypatch.undo()
        importlib.reload(invariance)


def test_exact_everywhere_else_with_the_difference_measured(monkeypatch):
    monkeypatch.setattr(invariance, "IN_IMAGE_CPU", False)
    invariance.matches_solo(REF.copy(), REF, "same bits")
    with pytest.raises(AssertionError, match=r"not bit-identical .*max \|diff\| .*max relative .*1 of 64 values differ"):
        invariance.matches_solo(np.where(np.arange(64) == 5, ulps(REF, 3), REF), REF, "3 ulp")
    with pytest.raises(AssertionError, match="not bit-identical"):
        invariance.exact(ulps(REF, 1), REF, "mechanics")


def test_image_cpu_warns_within_tolerance(monkeypatch):
    monkeypatch.setattr(invariance, "IN_IMAGE_CPU", True)
    with pytest.warns(invariance.NotBitIdentical, match="within the float32 tolerance"):
        invariance.matches_solo(ulps(REF, 5), REF, "5 ulp", [("other", OTHER)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        invariance.matches_solo(REF.copy(), REF, "same bits", [("other", OTHER)])     # no warning when identical


def test_image_cpu_still_fails_beyond_tolerance_and_on_swaps(monkeypatch):
    monkeypatch.setattr(invariance, "IN_IMAGE_CPU", True)
    with pytest.raises(AssertionError, match="beyond the float32 tolerance"):
        invariance.matches_solo(REF * np.float32(1 + 1e-4), REF, "1e-4 relative")
    with pytest.raises(AssertionError, match="beyond the float32 tolerance"):
        invariance.matches_solo(OTHER.copy(), REF, "swapped", [("other", OTHER)])
    near = ulps(REF, 2)                                       # within tolerance of REF, but it is exactly `near`'s
    with pytest.raises(AssertionError, match="another request's output"):
        invariance.matches_solo(near.copy(), REF, "closer to another", [("near", near)])
    with pytest.raises(AssertionError, match="shapes differ"):
        invariance.matches_solo(REF[:32], REF, "shape")
    with pytest.raises(AssertionError, match="not bit-identical"):
        invariance.exact(ulps(REF, 1), REF, "mechanics stay exact")


def test_bfloat16_computations_use_its_tolerance(monkeypatch):
    monkeypatch.setattr(invariance, "IN_IMAGE_CPU", True)
    got = REF * np.float32(1 + 1e-3)
    with pytest.warns(invariance.NotBitIdentical):
        invariance.matches_solo(got, REF, "bf16", [("other", OTHER)], "bfloat16")
    with pytest.raises(AssertionError, match="beyond the float32 tolerance"):
        invariance.matches_solo(got, REF, "f32")


def test_all_failing_cases_are_reported_together(monkeypatch):
    monkeypatch.setattr(invariance, "IN_IMAGE_CPU", False)
    with pytest.raises(AssertionError, match=r"2 of 3 checks failed:\nfirst: .*\nthird: "):
        invariance.all_match_solo([(ulps(REF, 1), REF, "first"), (REF, REF, "second"), (ulps(REF, 2), REF, "third")])
