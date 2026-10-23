"""Task 3: Prefix Scan - serial vs. 2-pass chunked parallel scan."""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230109011
N = 2_000_000
REPEATS = 5            # median of N runs to reduce timing noise


def workers_count():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    return os.cpu_count() or 1


def serial_scan(a):
    """Inclusive prefix sum: out[i] = out[i-1] + a[i]."""
    return np.cumsum(a, dtype=np.int64)


def parallel_scan(a, workers):
    n = len(a)
    step = -(-n // workers)
    bounds = [(i, min(i + step, n)) for i in range(0, n, step)]
    out = np.empty(n, dtype=np.int64)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        # Pass 1: local inclusive scan of each chunk (chunk total = last element)
        def local(b):
            lo, hi = b
            np.cumsum(a[lo:hi], dtype=np.int64, out=out[lo:hi])
        list(ex.map(local, bounds))

        # Offset reduction: exclusive scan over chunk totals (tiny, done serially)
        totals = np.array([out[hi - 1] for _, hi in bounds], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(totals)[:-1]))

        # Pass 2: add each chunk's offset
        def apply(args):
            (lo, hi), off = args
            out[lo:hi] += off
        list(ex.map(apply, zip(bounds, offsets)))
    return out


def timed(fn):
    runs = []
    result = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        result = fn()
        runs.append((time.perf_counter() - t0) * 1000.0)
    runs.sort()
    return result, runs[len(runs) // 2]


def main():
    rng = np.random.default_rng(STUDENT_ID)
    a = rng.integers(1, 11, size=N, dtype=np.int64)
    workers = workers_count()

    serial_scan(a)
    parallel_scan(a, workers)            # warm-up

    s_out, t_s = timed(lambda: serial_scan(a))
    p_out, t_p = timed(lambda: parallel_scan(a, workers))

    ok = bool(p_out[-1] == s_out[-1]) and bool(np.array_equal(p_out, s_out))
    status = "PASS" if ok else "FAIL"

    print(f"Student ID : {STUDENT_ID} | N = {N:,} | workers = {workers} (median of {REPEATS})")
    print(f"Last element: serial = {s_out[-1]} | parallel = {p_out[-1]}")
    print(f"Serial Time: {t_s:.2f} ms | Parallel Time: {t_p:.2f} ms | "
          f"Speedup: {t_s / t_p:.2f}x | Verification: {status}")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
