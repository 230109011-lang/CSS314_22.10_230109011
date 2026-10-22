"""Task 5: The overhead crossover point.

Benchmarks y[i] = a * x[i] + b with a single thread vs. a thread pool using
all logical cores, for N = 100 ... 10,000,000, then reports programmatically
the range of N where multi-threading flips from SLOWER to FASTER than serial.

Both versions work in place (np.multiply / np.add with out=) so neither
allocates memory inside the timed region. The pool is created once and reused;
its dispatch cost (submit + wait for every chunk) IS part of each timed call.
NumPy releases the GIL on array operations, so the threads run in parallel.

Usage:
    python task5.py
"""
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230109011"
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
A, B = 2.5, 1.0
TRIALS = 7            # independent timing trials per (N, mode); median is used
REFINE_POINTS = 12    # extra log-spaced sizes tried inside the crossover bracket


def reps_for(n: int) -> int:
    """Operations per trial: many for tiny arrays, few for huge ones."""
    return max(5, min(2000, 20_000_000 // n))


def serial(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def chunk(x, y, lo, hi):
    xs, ys = x[lo:hi], y[lo:hi]
    np.multiply(xs, A, out=ys)
    np.add(ys, B, out=ys)


def parallel(pool, workers, x, y):
    n = len(x)
    w = min(workers, n)
    bounds = [n * i // w for i in range(w + 1)]
    futures = [pool.submit(chunk, x, y, bounds[i], bounds[i + 1]) for i in range(w)]
    for f in futures:
        f.result()


def time_per_call(fn, reps: int) -> float:
    """Median (over TRIALS) of the average seconds per call."""
    fn()  # warm-up
    per_call = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        per_call.append((time.perf_counter() - t0) / reps)
    return statistics.median(per_call)


def measure(pool, workers, n: int, rng) -> tuple:
    x = rng.random(n)
    y_ser = np.empty(n)
    y_par = np.empty(n)
    reps = reps_for(n)
    t_ser = time_per_call(lambda: serial(x, y_ser), reps)
    t_par = time_per_call(lambda: parallel(pool, workers, x, y_par), reps)
    if not np.array_equal(y_ser, y_par):
        raise RuntimeError(f"parallel result differs from serial at N={n}")
    return t_ser, t_par


def fmt_time(seconds: float) -> str:
    us = seconds * 1e6
    return f"{us:,.1f} us" if us < 1000 else f"{us / 1000:,.3f} ms"


def print_table(rows) -> None:
    header = f"{'N':>12} | {'Serial':>12} | {'Parallel':>12} | {'Speedup':>8} | Winner"
    print(header)
    print("-" * (len(header) + 8))
    for n, t_ser, t_par in rows:
        sp = t_ser / t_par
        print(f"{n:>12,} | {fmt_time(t_ser):>12} | {fmt_time(t_par):>12} | {sp:>7.2f}x | "
              f"{'PARALLEL' if sp > 1.0 else 'serial'}")


def find_bracket(rows):
    """Smallest N from which parallel stays faster for every larger N tested.

    Returns (last_slower_n, first_faster_n); None for a missing side.
    """
    faster = [t_par < t_ser for _, t_ser, t_par in rows]
    k = None
    for i in range(len(rows)):
        if all(faster[i:]):
            k = i
            break
    if k is None:
        return rows[-1][0], None            # never (stably) faster
    if k == 0:
        return None, rows[0][0]             # faster everywhere
    return rows[k - 1][0], rows[k][0]


def main() -> None:
    workers = os.cpu_count() or 1
    rng = np.random.default_rng(int("".join(c for c in STUDENT_ID if c.isdigit())) % 2**32)

    print(f"Student ID    : {STUDENT_ID}")
    print(f"Platform      : {platform.platform()}")
    print(f"Logical cores : {workers}  (thread pool size = {workers})")
    print(f"Operation     : y[i] = {A} * x[i] + {B}  (float64, in place)")
    print(f"Timing        : median of {TRIALS} trials, dispatch overhead included")
    print()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [(n, *measure(pool, workers, n, rng)) for n in SIZES]
        print("Required sizes")
        print_table(rows)

        lo, hi = find_bracket(rows)
        print()
        if hi is None:
            print(f"RESULT: multi-threading was never reliably faster than serial "
                  f"up to N = {SIZES[-1]:,} on this hardware.")
            return
        if lo is None:
            print(f"RESULT: multi-threading was already faster than serial at the "
                  f"smallest size tested (N = {hi:,}); crossover is below N = {hi:,}.")
            return

        print(f"Coarse range: parallel SLOWER at N = {lo:,}, FASTER at N = {hi:,}")
        print(f"Refining between {lo:,} and {hi:,} ({REFINE_POINTS} log-spaced sizes)...")
        extra = sorted({int(v) for v in np.geomspace(lo, hi, REFINE_POINTS + 2)[1:-1]})
        fine_rows = [(n, *measure(pool, workers, n, rng)) for n in extra]
        print()
        print("Refinement sizes")
        print_table(fine_rows)

        all_rows = sorted(rows + fine_rows)
        lo2, hi2 = find_bracket(all_rows)

    print()
    print("=" * 70)
    if lo2 is not None and hi2 is not None:
        print(f"CROSSOVER: multi-threading is SLOWER than serial up to N = {lo2:,}")
        print(f"           and FASTER than serial from N = {hi2:,} onward.")
        print(f"           Tipping point lies in the range ({lo2:,}, {hi2:,}].")
    else:
        print("CROSSOVER: no stable crossover found in the refined sweep (measurement noise).")
    print("=" * 70)


if __name__ == "__main__":
    main()
