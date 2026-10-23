"""Task 4: IPC overhead - pickled pipe transfer (Path A) vs. zero-copy shared memory (Path B)."""
import multiprocessing as mp
import sys
import time
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230109011
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 5            # median of N runs to reduce timing noise


def worker(conn):
    """Long-lived consumer process (started once, so process start-up is not measured)."""
    while True:
        msg = conn.recv()
        if msg is None:
            break
        if isinstance(msg, tuple):                       # Path B: attach to shared memory
            name, n = msg
            shm = shared_memory.SharedMemory(name=name)
            view = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            last = float(view[-1])                       # touch data; no copy was made
            del view
            shm.close()
        else:                                            # Path A: array arrived pickled
            last = float(msg[-1])
        conn.send(last)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def bench(conn, n, rng):
    arr = rng.random(n)
    a, b_attach, b_total = [], [], []
    for _ in range(REPEATS):
        # Path A: pickle -> pipe -> unpickle in child -> ack
        t0 = time.perf_counter()
        conn.send(arr)
        got = conn.recv()
        a.append((time.perf_counter() - t0) * 1000.0)
        assert got == arr[-1]

        # Path B: shared memory
        t0 = time.perf_counter()
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        view = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
        view[:] = arr                                    # one-time write into the shared block
        t1 = time.perf_counter()
        conn.send((shm.name, n))                         # only a name crosses the pipe
        got = conn.recv()
        t2 = time.perf_counter()
        assert got == arr[-1]
        b_attach.append((t2 - t1) * 1000.0)
        b_total.append((t2 - t0) * 1000.0)
        del view
        shm.close()
        shm.unlink()
    return median(a), median(b_attach), median(b_total)


def main():
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=worker, args=(child,), daemon=True)
    p.start()

    rng = np.random.default_rng(STUDENT_ID)
    bench(parent, 100_000, rng)                          # warm-up

    print(f"Student ID: {STUDENT_ID} | float64 arrays | median of {REPEATS} runs")
    print(f"{'N':>12} | {'Size MB':>8} | {'A: pipe/pickle':>15} | {'B: shm attach':>14} | "
          f"{'B: shm + copy-in':>17} | {'A / B(attach)':>13}")
    for n in SIZES:
        a, b_attach, b_total = bench(parent, n, rng)
        print(f"{n:>12,} | {n * 8 / 1e6:>8.1f} | {a:>12.3f} ms | {b_attach:>11.3f} ms | "
              f"{b_total:>14.3f} ms | {a / b_attach:>12.1f}x")

    parent.send(None)
    p.join()


if __name__ == "__main__":
    main()
