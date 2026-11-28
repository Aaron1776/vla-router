"""The servers' request batcher (SPEC §3, §12; effort step 5): ordering, the wait window, capacity, error isolation
and concurrent clients, with a fake model. Pure Python; batch invariance of the real models is tested in
tests/reference/test_batch_invariance.py."""

import random
import threading
import time

import pytest

from vlar_harness.servers.batching import Batcher, BatchTooLarge, pad_rows


def spawn(b, key, payload, results, errors):
    def go():
        try:
            results[key] = b.submit(payload, timeout=10)
        except Exception as e:  # noqa: BLE001
            errors[key] = e
    t = threading.Thread(target=go)
    t.start()
    return t


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.001)


class Gate:
    """A run function whose first call blocks until released, so later requests queue up; records every batch."""

    def __init__(self, fn=lambda xs: [x * 10 for x in xs]):
        self.fn, self.batches, self.release = fn, [], threading.Event()

    def __call__(self, xs):
        if not self.batches:
            self.release.wait(5)
        self.batches.append(list(xs))
        return self.fn(xs)


def queue_behind_first(b, payloads, results, errors):
    """Submit payloads[0], wait until it is running (blocked in the gate), then queue the rest one by one."""
    threads = [spawn(b, 0, payloads[0], results, errors)]
    wait_for(lambda: b.stats.batches == 1)
    for i, p in enumerate(payloads[1:], start=1):
        threads.append(spawn(b, i, p, results, errors))
        wait_for(lambda i=i: len(b._queue) == i)
    return threads


def test_each_request_gets_its_own_result():
    b = Batcher("t", prepare=lambda p: p, run=lambda xs: [x * 10 for x in xs], capacity=(4,), wait_window_s=0.01)
    results, errors = {}, {}
    for t in [spawn(b, i, i, results, errors) for i in range(40)]:
        t.join()
    assert results == {i: i * 10 for i in range(40)} and not errors
    assert b.stats.requests == 40 and max(b.stats.sizes) <= 4
    b.close()


def test_the_wait_window_collects_requests_into_one_batch():
    gate = Gate()
    b = Batcher("t", prepare=lambda p: p, run=gate, capacity=(8,), wait_window_s=0.0)
    results, errors = {}, {}
    threads = queue_behind_first(b, [0, 1, 2, 3], results, errors)
    gate.release.set()
    for t in threads:
        t.join()
    assert gate.batches == [[0], [1, 2, 3]]          # everything waiting is taken together, in arrival order
    b.close()

    window = Batcher("w", prepare=lambda p: p, run=lambda xs: [len(xs)] * len(xs), capacity=(8,), wait_window_s=0.5)
    results = {}
    threads = [spawn(window, i, i, results, errors) for i in range(3)]
    for t in threads:
        t.join()
    assert results == {0: 3, 1: 3, 2: 3} and window.stats.sizes == [3]   # arrivals within the window: one batch
    t0 = time.monotonic()
    assert window.submit(9) == 1                      # a lone request waits out the window, then runs alone
    assert time.monotonic() - t0 >= 0.45
    window.close()


def test_capacity_splits_batches_and_nothing_overtakes():
    gate = Gate(lambda xs: [x["id"] for x in xs])
    b = Batcher("t", prepare=lambda p: p, run=gate, capacity=(2, 5), cost=lambda p: p["cost"], wait_window_s=0.0)
    payloads = [{"id": 0, "cost": (1, 1)}, {"id": 1, "cost": (1, 1)}, {"id": 2, "cost": (1, 1)},
                {"id": 3, "cost": (1, 5)}, {"id": 4, "cost": (1, 1)}, {"id": 5, "cost": (1, 2)}]
    results, errors = {}, {}
    threads = queue_behind_first(b, payloads, results, errors)
    gate.release.set()
    for t in threads:
        t.join()
    # [1, 2] fills the first component; 3 needs all 5 rows of the second; 4 and 5 fit together.
    assert [[x["id"] for x in batch] for batch in gate.batches] == [[0], [1, 2], [3], [4, 5]]
    assert results == {i: i for i in range(6)} and not errors
    with pytest.raises(BatchTooLarge, match="exceeds the fixed batch capacity"):
        b.submit({"id": 9, "cost": (1, 6)})
    b.close()


def test_one_bad_request_does_not_fail_its_batch_mates():
    def prepare(p):
        if p == "malformed":
            raise ValueError("malformed request")
        return p

    def run(xs):
        if "poison" in xs:
            raise RuntimeError("the model failed on a poison input")
        return [x.upper() for x in xs]

    gate = Gate(run)
    b = Batcher("t", prepare=prepare, run=gate, capacity=(8,), wait_window_s=0.0)
    results, errors = {}, {}
    threads = queue_behind_first(b, ["a", "b", "poison", "c"], results, errors)
    gate.release.set()
    for t in threads:
        t.join()
    with pytest.raises(ValueError, match="malformed"):
        b.submit("malformed")                         # fails in prepare, alone, before any batch
    assert gate.batches[:2] == [["a"], ["b", "poison", "c"]]          # the batch failed ...
    assert gate.batches[2:] == [["b"], ["poison"], ["c"]]             # ... and each request ran again alone
    assert results == {0: "A", 1: "B", 3: "C"}
    assert set(errors) == {2} and isinstance(errors[2], RuntimeError)
    assert b.stats.reruns_alone == 3
    b.close()


def test_many_concurrent_clients():
    sizes = []

    def run(xs):
        sizes.append(len(xs))
        time.sleep(0.002)
        return [(x, "done") for x in xs]

    b = Batcher("t", prepare=lambda p: p, run=run, capacity=(8,), wait_window_s=0.003)
    out, lock = [], threading.Lock()

    def client(w):
        rng = random.Random(w)
        for k in range(25):
            time.sleep(rng.uniform(0, 0.002))
            r = b.submit((w, k), timeout=10)
            with lock:
                out.append(r)

    ts = [threading.Thread(target=client, args=(w,)) for w in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(out) == sorted(((w, k), "done") for w in range(8) for k in range(25))
    assert max(sizes) <= 8 and sum(sizes) == 200 and len(sizes) < 200        # requests were batched
    b.close()
    with pytest.raises(RuntimeError, match="closed"):
        b.submit((0, 0))


def test_pad_rows_repeats_the_first_row_to_the_fixed_size():
    assert pad_rows(["a", "b"], 4) == ["a", "b", "a", "a"]
    assert pad_rows(["a"], 1) == ["a"]
    with pytest.raises(BatchTooLarge):
        pad_rows(["a", "b"], 1)
    with pytest.raises(ValueError):
        pad_rows([], 2)
