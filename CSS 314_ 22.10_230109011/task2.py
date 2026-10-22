"""Task 2: Multi-core scaling vs. Amdahl's Law.

Parallel summation of 20,000,000 float64 values using worker threads.
NumPy releases the GIL inside np.sum on large arrays, so threads give real
parallelism here (no data copying: each worker sums a view of its own slice).

Usage:
    python task2.py [STUDENT_ID]
"""
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230109011"
N = 20_000_000
P_PARALLEL = 0.95   # fraction of the program assumed parallelizable (Amdahl)
REPEATS = 7         # timed runs per worker count; the median is reported


def seed_from_id(student_id: str) -> int:
    digits = "".join(ch for ch in student_id if ch.isdigit())
    if digits:
        return int(digits) % (2**32)
    return sum((i + 1) * ord(c) for i, c in enumerate(student_id)) % (2**32)


def amdahl_bound(p: int, parallel_fraction: float = P_PARALLEL) -> float:
    """Theoretical max speedup: S(p) = 1 / ((1 - P) + P / p)."""
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p)


def chunk_sum(data: np.ndarray, lo: int, hi: int) -> float:
    return float(np.sum(data[lo:hi]))


def parallel_sum(pool: ThreadPoolExecutor, data: np.ndarray, p: int) -> float:
    n = len(data)
    bounds = [n * i // p for i in range(p + 1)]
    futures = [pool.submit(chunk_sum, data, bounds[i], bounds[i + 1]) for i in range(p)]
    return sum(f.result() for f in futures)


def benchmark(data: np.ndarray, p: int) -> tuple[float, float]:
    """Return (median wall time in ms, parallel sum) for p worker threads."""
    with ThreadPoolExecutor(max_workers=p) as pool:
        result = parallel_sum(pool, data, p)  # warm-up (threads start, caches fill)
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            result = parallel_sum(pool, data, p)
            times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times), result


def main() -> None:
    student_id = sys.argv[1] if len(sys.argv) > 1 else STUDENT_ID
    seed = seed_from_id(student_id)
    max_cores = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    print(f"Student ID    : {student_id} (seed {seed})")
    print(f"Platform      : {platform.platform()}")
    print(f"Logical cores : {max_cores}")
    print(f"Array size    : {N:,} float64 elements")
    print(f"Amdahl P      : {P_PARALLEL}  |  runs per row: {REPEATS} (median)")

    rng = np.random.default_rng(seed)
    data = rng.random(N, dtype=np.float64)
    reference = float(np.sum(data))

    rows = []
    t1 = None
    for p in worker_counts:
        wall_ms, result = benchmark(data, p)
        if p == 1:
            t1 = wall_ms
        speedup = t1 / wall_ms
        rows.append((p, wall_ms, speedup, amdahl_bound(p), result))

    print()
    header = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}"
    print(header)
    print("-" * len(header))
    for p, wall_ms, speedup, bound, _ in rows:
        print(f"{p:>6} | {wall_ms:>14.3f} | {speedup:>7.2f}x | {bound:>17.2f}x")

    worst = max(abs(r[4] - reference) / abs(reference) for r in rows)
    print()
    print(f"Serial reference sum : {reference:.6f}")
    print(f"Max relative diff    : {worst:.2e} (parallel vs serial sum)")
    print(f"Amdahl limit (p->inf): {1.0 / (1.0 - P_PARALLEL):.2f}x")


if __name__ == "__main__":
    main()
