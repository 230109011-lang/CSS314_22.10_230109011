import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Task 6: Synchronization cost and data loss.
 *
 * 4 threads each increment one shared counter 2,000,000 times:
 *   1. UNSYNCHRONIZED : plain int, no protection -> lost updates (race condition).
 *                       The read-modify-write is done through an opaque VarHandle
 *                       (no atomicity, no fences) so the JIT cannot fold the whole
 *                       loop into a single "+= 2,000,000", which would hide the race.
 *   2. AtomicInteger  : hardware atomic read-modify-write (lock-prefixed add / CAS).
 *   3. ReentrantLock  : mutual exclusion with a lock.
 *
 * Reports the expected vs. actual value, the percentage of lost updates, and
 * the wall-clock time penalty of each correct version over the racy one.
 *
 * Usage:  javac Task6.java && java Task6
 */
public class Task6 {
    static final String STUDENT_ID = "230109011";
    static final int THREADS = 4;
    static final int PER_THREAD = 2_000_000;
    static final long EXPECTED = (long) THREADS * PER_THREAD;
    static final int REPEATS = 3;   // runs per variant; median time reported

    // Shared state (reset before every run)
    static int plainCounter;                                   // deliberately unprotected
    static final VarHandle PLAIN;
    static {
        try {
            PLAIN = MethodHandles.lookup().findStaticVarHandle(Task6.class, "plainCounter", int.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }
    static final AtomicInteger atomicCounter = new AtomicInteger();
    static int lockedCounter;                                  // guarded by 'lock'
    static final ReentrantLock lock = new ReentrantLock();

    enum Variant {
        UNSYNC("Unsynchronized"),
        ATOMIC("AtomicInteger"),
        LOCK("ReentrantLock");

        final String label;
        Variant(String label) { this.label = label; }
    }

    static final class Result {
        final double ms;
        final long value;
        Result(double ms, long value) { this.ms = ms; this.value = value; }
    }

    static Runnable body(Variant v, int iterations) {
        switch (v) {
            case UNSYNC:
                return () -> {
                    for (int k = 0; k < iterations; k++) {
                        int cur = (int) PLAIN.getOpaque();     // read ...
                        PLAIN.setOpaque(cur + 1);              // ... modify, write: not atomic
                    }
                };
            case ATOMIC:
                return () -> {
                    for (int k = 0; k < iterations; k++) {
                        atomicCounter.incrementAndGet();
                    }
                };
            default:
                return () -> {
                    for (int k = 0; k < iterations; k++) {
                        lock.lock();
                        try {
                            lockedCounter++;
                        } finally {
                            lock.unlock();
                        }
                    }
                };
        }
    }

    static long readCounter(Variant v) {
        switch (v) {
            case UNSYNC: return plainCounter;
            case ATOMIC: return atomicCounter.get();
            default:     return lockedCounter;
        }
    }

    static void resetCounters() {
        plainCounter = 0;
        atomicCounter.set(0);
        lockedCounter = 0;
    }

    static Result run(Variant v, int iterations) throws Exception {
        resetCounters();
        CountDownLatch ready = new CountDownLatch(THREADS);
        CountDownLatch go = new CountDownLatch(1);
        Thread[] workers = new Thread[THREADS];
        for (int t = 0; t < THREADS; t++) {
            Runnable work = body(v, iterations);
            workers[t] = new Thread(() -> {
                ready.countDown();
                try {
                    go.await();
                } catch (InterruptedException e) {
                    return;
                }
                work.run();
            }, "worker-" + t);
            workers[t].start();
        }
        ready.await();
        long t0 = System.nanoTime();
        go.countDown();
        for (Thread w : workers) {
            w.join();
        }
        long t1 = System.nanoTime();
        return new Result((t1 - t0) / 1_000_000.0, readCounter(v));
    }

    static String n(long v) {
        return String.format(Locale.US, "%,d", v);
    }

    static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Student ID       : " + STUDENT_ID);
        System.out.println("OS               : " + System.getProperty("os.name") + " "
                + System.getProperty("os.version") + " (" + System.getProperty("os.arch") + ")");
        System.out.println("Java             : " + System.getProperty("java.version"));
        System.out.println("Logical cores    : " + Runtime.getRuntime().availableProcessors());
        System.out.println("Threads          : " + THREADS);
        System.out.println("Increments/thread: " + n(PER_THREAD));
        System.out.println("Runs per variant : " + REPEATS + " (median time reported)");
        System.out.println();

        // Warm-up so the JIT has compiled every loop before timing starts.
        for (Variant v : Variant.values()) {
            run(v, 200_000);
        }

        double[] medianMs = new double[Variant.values().length];
        long[] lastValue = new long[Variant.values().length];

        for (Variant v : Variant.values()) {
            double[] times = new double[REPEATS];
            System.out.println("--- " + v.label + " ---");
            for (int r = 0; r < REPEATS; r++) {
                Result res = run(v, PER_THREAD);
                times[r] = res.ms;
                lastValue[v.ordinal()] = res.value;
                long lost = EXPECTED - res.value;
                System.out.printf(Locale.US,
                        "  run %d: expected %s | actual %s | lost %s (%.4f%%) | %.2f ms%n",
                        r + 1, n(EXPECTED), n(res.value), n(lost),
                        100.0 * lost / EXPECTED, res.ms);
            }
            medianMs[v.ordinal()] = median(times);
            System.out.println();
        }

        double base = medianMs[Variant.UNSYNC.ordinal()];
        String header = String.format(Locale.US, "%-16s | %14s | %11s | %12s | %s",
                "Variant", "Wall Time (ms)", "Correct?", "Penalty (ms)", "Slowdown vs unsync");
        System.out.println(header);
        System.out.println("-".repeat(header.length() + 4));
        for (Variant v : Variant.values()) {
            double ms = medianMs[v.ordinal()];
            boolean correct = lastValue[v.ordinal()] == EXPECTED;
            System.out.printf(Locale.US, "%-16s | %14.2f | %11s | %12.2f | %.2fx%n",
                    v.label, ms, correct ? "yes" : "NO", ms - base, ms / base);
        }

        System.out.println();
        System.out.printf(Locale.US, "Synchronization penalty, AtomicInteger: %+.2f ms (%.2fx the unsynchronized time)%n",
                medianMs[Variant.ATOMIC.ordinal()] - base, medianMs[Variant.ATOMIC.ordinal()] / base);
        System.out.printf(Locale.US, "Synchronization penalty, ReentrantLock: %+.2f ms (%.2fx the unsynchronized time)%n",
                medianMs[Variant.LOCK.ordinal()] - base, medianMs[Variant.LOCK.ordinal()] / base);
    }
}
