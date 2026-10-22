"""Task 7: Arithmetic intensity vs. memory bottlenecks.

Kernel A (memory-bound):  B[i] = A[i]                         (vector copy)
Kernel B (compute-bound): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])

Both run on 30,000,000 float64 elements with 1, 2 and max logical cores
(thread pool; NumPy releases the GIL inside array operations). For each
kernel we print wall time, speedup S(p) = T(1) / T(p) and scaling efficiency
E(p) = S(p) / p, then state which kernel stalled because of memory
saturation.

Kernel B works block by block (64K elements) with per-thread scratch buffers
that stay in cache, so its cost is dominated by the trigonometry, not by RAM.

Usage:
    python task7.py
"""
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230109011"
N = 30_000_000
BLOCK = 1 << 16
PLATEAU_GAIN = 1.25       # < 1.25x extra speedup from the previous core count => "stalled"
STALL_EFFICIENCY = 0.60   # fallback rule when only two core counts exist


def copy_chunk(A, B, lo, hi, bufs):
    np.copyto(B[lo:hi], A[lo:hi])


def compute_chunk(A, B, lo, hi, bufs):
    t1, t2 = bufs
    for s in range(lo, hi, BLOCK):
        e = min(s + BLOCK, hi)
        n = e - s
        a, out = A[s:e], B[s:e]
        x, y = t1[:n], t2[:n]
        np.sin(a, out=x)
        np.cos(a, out=y)
        np.multiply(x, y, out=x)
        np.tan(a, out=y)
        np.add(x, y, out=out)


def run_parallel(pool, chunk_fn, A, B, workers, all_bufs):
    bounds = [N * i // workers for i in range(workers + 1)]
    futures = [pool.submit(chunk_fn, A, B, bounds[w], bounds[w + 1], all_bufs[w])
               for w in range(workers)]
    for f in futures:
        f.result()


def bench(chunk_fn, A, B, workers, trials):
    """Median wall time in ms for the kernel on `workers` threads."""
    all_bufs = [(np.empty(BLOCK), np.empty(BLOCK)) for _ in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        run_parallel(pool, chunk_fn, A, B, workers, all_bufs)   # warm-up
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            run_parallel(pool, chunk_fn, A, B, workers, all_bufs)
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def scaling_table(title, chunk_fn, A, B, counts, trials, bytes_moved=None):
    print(title)
    cols = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Efficiency':>10}"
    if bytes_moved:
        cols += f" | {'Bandwidth':>11}"
    print(cols)
    print("-" * len(cols))
    rows, t1 = [], None
    for p in counts:
        ms = bench(chunk_fn, A, B, p, trials)
        if p == 1:
            t1 = ms
        speedup = t1 / ms
        eff = speedup / p
        line = f"{p:>6} | {ms:>14.2f} | {speedup:>7.2f}x | {eff * 100:>9.1f}%"
        if bytes_moved:
            line += f" | {bytes_moved / (ms / 1000.0) / 1e9:>7.1f} GB/s"
        print(line)
        rows.append((p, ms, speedup, eff))
    print()
    return rows


def main() -> None:
    max_cores = os.cpu_count() or 1
    counts = sorted({1, 2, max_cores})

    print(f"Student ID    : {STUDENT_ID}")
    print(f"Platform      : {platform.platform()}")
    print(f"Logical cores : {max_cores}")
    print(f"Array size    : {N:,} float64 elements ({N * 8 / 1e6:.0f} MB per array)")
    print(f"Core counts   : {counts}")
    print()

    rng = np.random.default_rng(int(STUDENT_ID) % 2**32)
    A = rng.random(N)
    B = np.empty(N)

    # Kernel A reads A and writes B: 2 * 8 bytes per element.
    rows_a = scaling_table("Kernel A (memory-bound): B[i] = A[i]",
                           copy_chunk, A, B, counts, trials=7, bytes_moved=2 * 8 * N)
    assert np.array_equal(A, B), "copy kernel produced a wrong result"

    rows_b = scaling_table("Kernel B (compute-bound): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])",
                           compute_chunk, A, B, counts, trials=3)
    idx = slice(0, 1000)
    ref = np.sin(A[idx]) * np.cos(A[idx]) + np.tan(A[idx])
    assert np.allclose(B[idx], ref), "trig kernel produced a wrong result"

    eff_a, eff_b = rows_a[-1][3], rows_b[-1][3]
    sp_a, sp_b = rows_a[-1][2], rows_b[-1][2]
    bw_a = 2 * 8 * N / (rows_a[-1][1] / 1000.0) / 1e9
    p = counts[-1]

    def stalled(rows):
        """A kernel has stalled if adding cores no longer adds speedup."""
        if len(rows) >= 3:
            gain = rows[-1][2] / rows[-2][2]      # speedup gain from previous count to max
            return gain < PLATEAU_GAIN, gain
        return rows[-1][3] < STALL_EFFICIENCY, None

    stall_a, gain_a = stalled(rows_a)
    stall_b, gain_b = stalled(rows_b)

    print("=" * 76)
    print(f"Efficiency at {p} cores:  Kernel A (copy) = {eff_a * 100:.1f}%"
          f"   |   Kernel B (sin/cos/tan) = {eff_b * 100:.1f}%")
    if gain_a is not None:
        print(f"Extra speedup from {counts[-2]} to {p} cores:  Kernel A = {gain_a:.2f}x"
              f"   |   Kernel B = {gain_b:.2f}x   (stall if < {PLATEAU_GAIN:.2f}x)")
    if p == 1:
        print("VERDICT: only 1 core available, scaling cannot be assessed.")
    elif stall_a and not stall_b:
        print(f"VERDICT: Kernel A (vector copy) STALLED in scaling due to memory channel "
              f"saturation:\n         speedup stuck at {sp_a:.2f}x on {p} cores, bandwidth flat "
              f"at ~{bw_a:.1f} GB/s.\n         Kernel B (sin/cos/tan, compute-bound) kept "
              f"scaling ({sp_b:.2f}x on {p} cores).")
    elif stall_a and stall_b:
        worse = "A (vector copy)" if eff_a < eff_b else "B (sin/cos/tan)"
        print(f"VERDICT: both kernels stopped scaling; Kernel {worse} is the more "
              f"memory-limited one (lower efficiency).")
    elif stall_b:
        print(f"VERDICT: Kernel B stalled, Kernel A did not; this is not memory saturation "
              f"(Kernel A speedup {sp_a:.2f}x).")
    else:
        print("VERDICT: no memory-channel saturation observed; both kernels kept scaling.")
    if p > 2 and eff_b < 0.9:
        print("NOTE: efficiency below 100% on the compute-bound kernel is expected when the "
              "logical\n      cores include SMT/hyper-threads (two threads share one physical core).")
    print("=" * 76)


if __name__ == "__main__":
    main()
