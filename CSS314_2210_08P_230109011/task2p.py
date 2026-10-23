"""Task 2: Pseudo-Random Generator State Bottleneck (Monte Carlo pi)."""
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230109011
TOTAL = 40_000_000
THREADS = 4
BATCH = 100            # samples drawn per generator call (same for both modes)
REPEATS = 5            # median of N runs to reduce timing noise


def count_hits(draw, n):
    """Draw n points in batches via `draw(k) -> (x, y)` and count those inside the unit circle."""
    hits = 0
    done = 0
    while done < n:
        k = min(BATCH, n - done)
        x, y = draw(k)
        hits += int(np.count_nonzero(x * x + y * y <= 1.0))
        done += k
    return hits


def run_shared():
    """4 threads, ONE generator; every draw must hold the mutex."""
    rng = np.random.default_rng(STUDENT_ID * 10)
    lock = threading.Lock()

    def draw(k):
        with lock:                       # generator state is shared -> serialized
            return rng.random(k), rng.random(k)

    return run(lambda i, n: count_hits(draw, n))


def run_independent():
    """4 threads, each with its own generator seeded STUDENT_ID*10 + thread_index; no locking."""
    def work(i, n):
        rng = np.random.default_rng(STUDENT_ID * 10 + i)
        return count_hits(lambda k: (rng.random(k), rng.random(k)), n)

    return run(work)


def run(work):
    per = TOTAL // THREADS
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        hits = sum(ex.map(lambda i: work(i, per), range(THREADS)))
    ms = (time.perf_counter() - t0) * 1000.0
    return 4.0 * hits / (per * THREADS), ms


def main():
    run_independent()                    # warm-up
    run_shared()

    def median_run(fn):
        runs = [fn() for _ in range(REPEATS)]
        runs.sort(key=lambda r: r[1])
        return runs[len(runs) // 2]      # (pi, ms) of the median-time run

    pi_s, t_s = median_run(run_shared)
    pi_i, t_i = median_run(run_independent)

    print(f"Student ID           : {STUDENT_ID}")
    print(f"Samples / threads    : {TOTAL:,} / {THREADS} (batch={BATCH}, median of {REPEATS})")
    print(f"Shared PRNG (mutex)  : pi = {pi_s:.8f} | error = {abs(pi_s - math.pi):.8f} | {t_s:.2f} ms")
    print(f"Independent PRNGs    : pi = {pi_i:.8f} | error = {abs(pi_i - math.pi):.8f} | {t_i:.2f} ms")
    print(f"Speedup (shared / independent): {t_s / t_i:.2f}x")


if __name__ == "__main__":
    main()
