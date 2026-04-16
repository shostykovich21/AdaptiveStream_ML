package com.adaptivestream;

/**
 * Smooths interval transitions to prevent state corruption during window resize.
 *
 * Problem: If interval suddenly drops from 10s to 1s mid-window, events buffered
 * for the 10s window may be split or duplicated. Spark's watermark advances based
 * on the trigger interval — a sudden change can cause late events to be dropped.
 *
 * Solution: Exponential smoothing with configurable rate limit.
 * - Interval can only change by at most MAX_CHANGE_RATIO per update cycle
 * - Shrinking is slower than expanding (conservative on latency-sensitive direction)
 * - Transitions complete over multiple cycles, giving Spark's state store time to adapt
 */
public class IntervalSmoother {

    private static final double MAX_SHRINK_RATIO = 0.5;   // can halve per cycle at most
    private static final double MAX_EXPAND_RATIO = 2.0;   // can double per cycle at most
    private static final double SMOOTHING_ALPHA = 0.3;     // EMA smoothing factor

    private long currentSmoothed;
    private boolean initialized = false;

    public IntervalSmoother() {
        this.currentSmoothed = -1;
    }

    /**
     * Takes a raw target interval and returns a smoothed, rate-limited interval.
     */
    public synchronized long smooth(long targetMs) {
        if (!initialized || currentSmoothed <= 0) {
            currentSmoothed = targetMs;
            initialized = true;
            return targetMs;
        }

        // EMA smoothing
        long smoothed = (long)(SMOOTHING_ALPHA * targetMs + (1 - SMOOTHING_ALPHA) * currentSmoothed);

        // Rate limiting
        long minAllowed = (long)(currentSmoothed * MAX_SHRINK_RATIO);
        long maxAllowed = (long)(currentSmoothed * MAX_EXPAND_RATIO);

        smoothed = Math.max(smoothed, minAllowed);
        smoothed = Math.min(smoothed, maxAllowed);

        // Floor/ceiling
        smoothed = Math.max(smoothed, 100);    // never below 100ms
        smoothed = Math.min(smoothed, 30000);  // never above 30s

        currentSmoothed = smoothed;
        return smoothed;
    }

    public long getCurrentSmoothed() {
        return currentSmoothed;
    }

    /**
     * Reset smoother (e.g., when entering fallback mode).
     */
    public synchronized void reset(long newBaseline) {
        currentSmoothed = newBaseline;
        initialized = true;
    }
}
