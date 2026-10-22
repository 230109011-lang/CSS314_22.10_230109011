"""Task 4: Work stealing and scheduling under skewed workloads.

1,000 items, item i costs sleep(0.0001 * (i % 20)) seconds. Four workers
process them with:
  A) static scheduling   - equal contiguous blocks, one per worker
  B) dynamic scheduling  - shared queue of chunks, workers pull the next chunk
                           when they are free (chunk size 1 and 5)

Per run we log total wall time and the finish-time spread between the first
and the last worker to finish. sleep() releases the GIL, so threads overlap
like real parallel workers.

Usage:
    python task4.py
"""
import os
import platform
import queue
import statistics
import threading
import time

ITEMS = 1000
WORKERS = 4
REPEATS = 3          # runs per strategy; the run with the median total time is shown
STUDENT_ID = "230109011"


def cost(i: int) -> float:
    """Simulated work for item i, in seconds."""
    return 0.0001 * (i % 20)


def process(i: int) -> None:
    time.sleep(cost(i))


def run_static() -> dict:
    """Strategy A: split [0, ITEMS) into WORKERS equal contiguous blocks."""
    block = ITEMS // WORKERS
    ranges = [(w * block, ITEMS if w == WORKERS - 1 else (w + 1) * block) for w in range(WORKERS)]
    finish = [0.0] * WORKERS
    counts = [0] * WORKERS
    barrier = threading.Barrier(WORKERS + 1)
    t0_holder = [0.0]

    def worker(w: int) -> None:
        lo, hi = ranges[w]
        barrier.wait()
        for i in range(lo, hi):
            process(i)
            counts[w] += 1
        finish[w] = time.perf_counter() - t0_holder[0]

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(WORKERS)]
    for t in threads:
        t.start()
    t0_holder[0] = time.perf_counter()
    barrier.wait()
    for t in threads:
        t.join()
    return summarize(finish, counts)


def run_dynamic(chunk: int) -> dict:
    """Strategy B: workers repeatedly take the next chunk from a shared queue."""
    work = queue.Queue()
    for lo in range(0, ITEMS, chunk):
        work.put((lo, min(lo + chunk, ITEMS)))

    finish = [0.0] * WORKERS
    counts = [0] * WORKERS
    barrier = threading.Barrier(WORKERS + 1)
    t0_holder = [0.0]

    def worker(w: int) -> None:
        barrier.wait()
        while True:
            try:
                lo, hi = work.get_nowait()
            except queue.Empty:
                break
            for i in range(lo, hi):
                process(i)
                counts[w] += 1
        finish[w] = time.perf_counter() - t0_holder[0]

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(WORKERS)]
    for t in threads:
        t.start()
    t0_holder[0] = time.perf_counter()
    barrier.wait()
    for t in threads:
        t.join()
    return summarize(finish, counts)


def summarize(finish: list, counts: list) -> dict:
    return {
        "total_ms": max(finish) * 1000.0,
        "first_ms": min(finish) * 1000.0,
        "last_ms": max(finish) * 1000.0,
        "delta_ms": (max(finish) - min(finish)) * 1000.0,
        "finish_ms": [f * 1000.0 for f in finish],
        "counts": counts,
    }


def median_run(fn, *args) -> dict:
    runs = [fn(*args) for _ in range(REPEATS)]
    runs.sort(key=lambda r: r["total_ms"])
    return runs[len(runs) // 2]


def main() -> None:
    total_work_ms = sum(cost(i) for i in range(ITEMS)) * 1000.0
    ideal_ms = total_work_ms / WORKERS

    print(f"Student ID    : {STUDENT_ID}")
    print(f"Platform      : {platform.platform()}")
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Items         : {ITEMS} (cost = 0.0001 * (i % 20) s)")
    print(f"Workers       : {WORKERS}")
    print(f"Runs/strategy : {REPEATS} (run with median total time shown)")
    print(f"Total work    : {total_work_ms:.1f} ms  ->  ideal with {WORKERS} workers: {ideal_ms:.1f} ms")
    print()

    strategies = [
        ("A: Static blocks", run_static, ()),
        ("B: Dynamic chunk=1", run_dynamic, (1,)),
        ("B: Dynamic chunk=5", run_dynamic, (5,)),
    ]

    # Warm-up so thread start-up costs do not land on the first strategy.
    run_dynamic(50)

    results = [(name, median_run(fn, *args)) for name, fn, args in strategies]

    header = (f"{'Strategy':<20} | {'Total (ms)':>10} | {'First done':>10} | "
              f"{'Last done':>10} | {'Delta (ms)':>10} | {'vs ideal':>8}")
    print(header)
    print("-" * len(header))
    for name, r in results:
        print(f"{name:<20} | {r['total_ms']:>10.2f} | {r['first_ms']:>10.2f} | "
              f"{r['last_ms']:>10.2f} | {r['delta_ms']:>10.2f} | {r['total_ms'] / ideal_ms:>7.2f}x")

    print()
    print("Per-worker detail (finish time in ms / items processed):")
    for name, r in results:
        parts = [f"W{w}: {r['finish_ms'][w]:.1f} / {r['counts'][w]}" for w in range(WORKERS)]
        print(f"  {name:<20} " + "  ".join(parts))


if __name__ == "__main__":
    main()
