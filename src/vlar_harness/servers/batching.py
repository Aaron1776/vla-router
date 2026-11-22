"""Dynamic request batching for the model servers (docs/SPEC.md §3, §12; amendment 24). Pure Python.

Each endpoint has one `Batcher`: a request queue and one worker thread that owns the model. A request handler calls
`submit(payload)`, which prepares the request in the caller's thread (decoding, the reference preprocessing; a bad
request fails here, alone) and waits for its result. The worker takes the oldest waiting request, keeps collecting
requests for up to `wait_window_s` while they fit the endpoint's capacity, and runs them as one batch.

Batch invariance is the run function's job, not the queue's: it pads every batch to the endpoint's fixed size and
draws each request's noise from that request's own seeds, so a request's result does not depend on which requests
share its batch, how many there are, or where it sits. That is also what makes error isolation exact: if a batch
raises, each of its requests is run again alone (a batch of one, padded to the same fixed size, so the same
computation), and only the requests that fail alone fail.

Capacity: each prepared request has a cost vector (for example one encoder row and K denoising rows for an X-VLA
`/sample`) and the endpoint a capacity vector; a batch never exceeds the capacity in any component, and a request
that alone exceeds it is refused at submit.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


class BatchTooLarge(ValueError):
    """A single request needs more than the endpoint's fixed batch capacity."""


@dataclass
class _Item:
    prepared: Any
    cost: tuple[int, ...]
    future: Future = field(default_factory=Future)
    arrived: float = 0.0


@dataclass
class BatchStats:
    batches: int = 0
    requests: int = 0
    reruns_alone: int = 0
    sizes: list = field(default_factory=list)     # requests per batch, in order


class Batcher:
    """`prepare(payload) -> prepared`, `cost(prepared) -> tuple`, `run(list[prepared]) -> list[result]` (one result
    per request, in order). `run` is called only from the worker thread."""

    def __init__(self, name: str, *, prepare: Callable[[Any], Any], run: Callable[[list], Sequence[Any]],
                 capacity: Sequence[int], cost: Callable[[Any], Sequence[int]] = lambda _: (1,),
                 wait_window_s: float = 0.005, clock: Callable[[], float] = time.monotonic):
        if not capacity or any(int(c) < 1 for c in capacity):
            raise ValueError(f"capacity must be positive, got {capacity}")
        if wait_window_s < 0:
            raise ValueError("wait_window_s must be >= 0")
        self.name = name
        self.prepare, self.run, self.cost = prepare, run, cost
        self.capacity = tuple(int(c) for c in capacity)
        self.wait_window_s = float(wait_window_s)
        self.clock = clock
        self.stats = BatchStats()
        self._queue: deque[_Item] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(target=self._loop, name=f"batcher-{name}", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ clients

    def submit(self, payload: Any, timeout: Optional[float] = None) -> Any:
        """Prepare in the calling thread, queue, wait for the result (or the request's own exception)."""
        prepared = self.prepare(payload)
        cost = tuple(int(c) for c in self.cost(prepared))
        if len(cost) != len(self.capacity):
            raise ValueError(f"cost {cost} does not match capacity {self.capacity}")
        if any(c > cap for c, cap in zip(cost, self.capacity)):
            raise BatchTooLarge(f"{self.name}: a request needing {cost} exceeds the fixed batch capacity "
                                f"{self.capacity}")
        item = _Item(prepared, cost, arrived=self.clock())
        with self._cv:
            if self._closed:
                raise RuntimeError(f"{self.name}: batcher is closed")
            self._queue.append(item)
            self._cv.notify_all()
        return item.future.result(timeout=timeout)

    def close(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._worker.join(timeout)

    # ------------------------------------------------------------ worker

    def _fits(self, used: list[int], cost: tuple[int, ...]) -> bool:
        return all(u + c <= cap for u, c, cap in zip(used, cost, self.capacity))

    def _take_batch(self) -> list[_Item]:
        """Oldest request first, then later ones (in arrival order) while they fit, until the window closes or the
        batch is full. A request that does not fit waits for the next batch, and nothing overtakes it."""
        with self._cv:
            while not self._queue and not self._closed:
                self._cv.wait()
            if not self._queue:
                return []
            first = self._queue.popleft()
            batch, used = [first], list(first.cost)
            deadline = first.arrived + self.wait_window_s
            while True:
                while self._queue and self._fits(used, self._queue[0].cost):
                    item = self._queue.popleft()
                    batch.append(item)
                    used = [u + c for u, c in zip(used, item.cost)]
                full = bool(self._queue) or any(u >= cap for u, cap in zip(used, self.capacity))
                remaining = deadline - self.clock()
                if full or remaining <= 0 or self._closed:
                    return batch
                self._cv.wait(remaining)

    def _run(self, items: list[_Item]) -> list[tuple[bool, Any]]:
        out = self.run([it.prepared for it in items])
        if len(out) != len(items):
            raise RuntimeError(f"{self.name}: run returned {len(out)} results for {len(items)} requests")
        return [(True, r) for r in out]

    def _loop(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                return
            self.stats.batches += 1
            self.stats.requests += len(batch)
            self.stats.sizes.append(len(batch))
            try:
                results = self._run(batch)
            except Exception as e:  # noqa: BLE001 - isolate the failing request(s)
                if len(batch) == 1:
                    results = [(False, e)]
                else:
                    results = []
                    for it in batch:                 # alone: the same fixed-size computation, so the same result
                        self.stats.reruns_alone += 1
                        try:
                            results.extend(self._run([it]))
                        except Exception as e1:  # noqa: BLE001
                            results.append((False, e1))
            for it, (ok, value) in zip(batch, results):
                if ok:
                    it.future.set_result(value)
                else:
                    it.future.set_exception(value)


def pad_rows(rows: list, size: int) -> list:
    """`rows` padded to exactly `size` entries by repeating the first row (a real input, so padding rows compute
    nothing unusual); the caller keeps only the first len(rows) outputs."""
    if not rows:
        raise ValueError("cannot pad an empty batch")
    if len(rows) > size:
        raise BatchTooLarge(f"{len(rows)} rows exceed the fixed batch size {size}")
    return list(rows) + [rows[0]] * (size - len(rows))
