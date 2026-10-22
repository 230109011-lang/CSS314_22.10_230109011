import java.io.InputStream;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Task 3: Cache line contention (false sharing).
 *
 * N threads (N = physical cores) each increment their own slot of a shared
 * long[] 50,000,000 times:
 *   1. UNPADDED : slots are adjacent (8 bytes apart) -> several threads hit
 *                 the same 64-byte cache line -> constant invalidation.
 *   2. PADDED   : slots are 128 bytes apart (>= one cache line, and also
 *                 covers the adjacent-line prefetcher) -> no sharing.
 *
 * Slots are accessed with VarHandle opaque get/set so the JIT cannot keep
 * the counter in a register and silently remove the memory traffic we want
 * to measure. (A plain local-variable counter is not used for the comparison:
 * the JIT folds "c++ in a loop" into a single add, which would measure
 * nothing.)
 *
 * Usage:  javac Task3.java && java Task3 [THREADS]
 */
public class Task3 {
    static final String STUDENT_ID = "230109011";
    static final long ITERATIONS = 50_000_000L;
    static final long WARMUP_ITERATIONS = 5_000_000L;
    static final int PAD_LONGS = 16;   // 16 * 8 bytes = 128-byte stride
    static final int REPEATS = 3;      // timed runs per mode; median reported

    static final VarHandle SLOT = MethodHandles.arrayElementVarHandle(long[].class);

    enum Mode {
        UNPADDED("Unpadded Shared"),
        PADDED("Padded (128 B)");

        final String label;
        Mode(String label) { this.label = label; }
    }

    /** Runs one benchmark pass and returns the wall time in milliseconds. */
    static double runOnce(Mode mode, int threads, long iterations) throws Exception {
        final int stride = (mode == Mode.PADDED) ? PAD_LONGS : 1;
        final long[] sharedCounts = new long[threads * stride];

        final CountDownLatch ready = new CountDownLatch(threads);
        final CountDownLatch go = new CountDownLatch(1);
        Thread[] workers = new Thread[threads];

        for (int t = 0; t < threads; t++) {
            final int idx = t * stride;
            workers[t] = new Thread(() -> {
                ready.countDown();
                try {
                    go.await();
                } catch (InterruptedException e) {
                    return;
                }
                for (long k = 0; k < iterations; k++) {
                    SLOT.setOpaque(sharedCounts, idx, (long) SLOT.getOpaque(sharedCounts, idx) + 1L);
                }
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

        for (int t = 0; t < threads; t++) {
            if (sharedCounts[t * stride] != iterations) {
                throw new IllegalStateException("Wrong count in slot " + t + ": " + sharedCounts[t * stride]);
            }
        }
        return (t1 - t0) / 1_000_000.0;
    }

    static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }

    // ---- physical core detection (Java only exposes logical processors) ----

    static String exec(String... cmd) throws Exception {
        Process p = new ProcessBuilder(cmd).redirectErrorStream(false).start();
        String out;
        try (InputStream in = p.getInputStream()) {
            out = new String(in.readAllBytes());
        }
        p.waitFor(10, TimeUnit.SECONDS);
        return out;
    }

    static int physicalCores(int logical) {
        String os = System.getProperty("os.name").toLowerCase();
        try {
            int n;
            if (os.contains("win")) {
                String out = exec("powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum");
                n = Integer.parseInt(out.trim());
            } else if (os.contains("mac")) {
                n = Integer.parseInt(exec("sysctl", "-n", "hw.physicalcpu").trim());
            } else {
                Set<String> cores = new HashSet<>();
                for (String line : exec("lscpu", "-p=CORE,SOCKET").split("\\R")) {
                    if (!line.isBlank() && !line.startsWith("#")) {
                        cores.add(line.trim());
                    }
                }
                n = cores.size();
            }
            return n > 0 ? n : logical;
        } catch (Exception e) {
            return logical;
        }
    }

    public static void main(String[] args) throws Exception {
        int logical = Runtime.getRuntime().availableProcessors();
        int physical = physicalCores(logical);
        int threads = args.length > 0 ? Integer.parseInt(args[0]) : physical;

        System.out.println("Student ID       : " + STUDENT_ID);
        System.out.println("OS               : " + System.getProperty("os.name") + " "
                + System.getProperty("os.version") + " (" + System.getProperty("os.arch") + ")");
        System.out.println("Java             : " + System.getProperty("java.version"));
        System.out.println("Logical cores    : " + logical);
        System.out.println("Physical cores   : " + physical);
        System.out.println("Threads used     : " + threads);
        System.out.println("Increments/thread: " + String.format("%,d", ITERATIONS));
        System.out.println("Runs per mode    : " + REPEATS + " (median reported)");
        System.out.println();

        // Warm-up so the JIT has compiled every loop before timing starts.
        for (Mode m : Mode.values()) {
            runOnce(m, threads, WARMUP_ITERATIONS);
        }

        double[] medians = new double[Mode.values().length];
        for (Mode m : Mode.values()) {
            double[] times = new double[REPEATS];
            for (int r = 0; r < REPEATS; r++) {
                times[r] = runOnce(m, threads, ITERATIONS);
            }
            medians[m.ordinal()] = median(times);
            StringBuilder runs = new StringBuilder();
            for (double t : times) {
                runs.append(String.format("%.1f ", t));
            }
            System.out.printf("  %-16s runs (ms): %s%n", m.label, runs.toString().trim());
        }

        double padded = medians[Mode.PADDED.ordinal()];
        System.out.println();
        String header = String.format("%-16s | %14s | %s", "Mode", "Wall Time (ms)", "Ratio vs Padded");
        System.out.println(header);
        System.out.println("-".repeat(header.length()));
        for (Mode m : Mode.values()) {
            double ms = medians[m.ordinal()];
            System.out.printf("%-16s | %14.3f | %14.2fx%n", m.label, ms, ms / padded);
        }

        System.out.println();
        System.out.printf("Execution ratio, Unpadded / Padded: %.2fx%n",
                medians[Mode.UNPADDED.ordinal()] / padded);
    }
}
