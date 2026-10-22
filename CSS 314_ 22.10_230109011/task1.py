"""Task 1: Hardware probe and serial baseline.

Usage:
    python task1.py [STUDENT_ID]

The Student ID is used as the RNG seed. Set STUDENT_ID below or pass it as
the first command-line argument.
"""
import os
import platform
import sys
import time

import numpy as np

STUDENT_ID = "230109011"  
N = 20_000_000


def cpu_model() -> str:
    """Best-effort CPU model string across platforms."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def seed_from_id(student_id: str) -> int:
    digits = "".join(ch for ch in student_id if ch.isdigit())
    if digits:
        return int(digits) % (2**32)
    # Non-numeric ID: derive a stable seed from the characters
    return sum((i + 1) * ord(c) for i, c in enumerate(student_id)) % (2**32)


def main() -> None:
    student_id = sys.argv[1] if len(sys.argv) > 1 else STUDENT_ID
    seed = seed_from_id(student_id)

    print(f"Student ID      : {student_id}")
    print(f"Seed            : {seed}")
    print(f"OS              : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform        : {platform.platform()}")
    print(f"CPU model       : {cpu_model()}")
    print(f"Logical cores   : {os.cpu_count()}")
    print(f"Python / NumPy  : {platform.python_version()} / {np.__version__}")
    print(f"Array size      : {N:,} float64 elements")

    
    rng = np.random.default_rng(seed)
    data = rng.random(N, dtype=np.float64)

    # Serial computation: one thread, sum then (population) standard deviation.
    # NumPy's sum/std reductions are single-threaded (no BLAS involved).
    start = time.perf_counter()
    total = float(np.sum(data))
    std = float(np.std(data))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"Sum             : {total:.6f}")
    print(f"Std deviation   : {std:.10f}")
    print(f"Serial time     : {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
