"""Task 1: Row-Major vs. Column-Major Parallel Cache Traversal."""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = int(os.environ.get("STUDENT_ID", "230109011"))  
N = 8000


def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    return os.cpu_count() or 1


def split(n, parts):
    step = -(-n // parts)
    return [(i, min(i + step, n)) for i in range(0, n, step)]


def sum_rows(m, lo, hi):
    """Mode A: each worker owns a block of rows, walks columns linearly."""
    s = 0.0
    for i in range(lo, hi):
        s += m[i, :].sum()          # contiguous memory
    return s


def sum_cols(m, lo, hi):
    """Mode B: each worker owns a block of columns, walks down rows (stride = N*8 bytes)."""
    s = 0.0
    for j in range(lo, hi):
        s += m[:, j].sum()          # strided: every element on a different cache line
    return s


def run(fn, m, workers):
    chunks = split(N, workers)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        total = sum(ex.map(lambda c: fn(m, *c), chunks))
    return (time.perf_counter() - t0) * 1000.0, total


def main():
    rng = np.random.default_rng(STUDENT_ID)
    m = rng.random((N, N), dtype=np.float64)   # C-order (row-major) layout
    workers = physical_cores()

    run(sum_rows, m, workers)                  # warm-up (page-in memory)

    ta, sa = run(sum_rows, m, workers)
    tb, sb = run(sum_cols, m, workers)

    print(f"Student ID          : {STUDENT_ID}")
    print(f"Logical cores       : {os.cpu_count()}")
    print(f"Workers (physical)  : {workers}")
    print(f"Matrix              : {N}x{N} float64 ({m.nbytes / 1e6:.0f} MB)")
    print(f"Mode A (Row-Major)  : {ta:.2f} ms  (sum={sa:.6f})")
    print(f"Mode B (Col-Major)  : {tb:.2f} ms  (sum={sb:.6f})")
    print(f"Cache penalty factor: {tb / ta:.2f}x  (Mode B / Mode A)")


if __name__ == "__main__":
    main()
