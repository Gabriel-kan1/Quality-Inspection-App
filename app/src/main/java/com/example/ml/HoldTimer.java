package com.example.ml;

/**
 * Tracks how long alignment has been held, so a frame only passes after the operator has
 * kept the phone steady for the full dwell rather than swinging through the tolerance band.
 *
 * <p>A single dropped frame doesn't restart the count. Feature matching is stochastic — a
 * hand tremor or a passing shadow can lose the homography for one frame out of a stream
 * that is otherwise solidly aligned, and resetting on that made the gate feel broken in
 * practice. Anything longer than the grace window is treated as the operator genuinely
 * losing alignment and does reset.
 *
 * <p>Time is passed in rather than read from a clock, which keeps this testable and lets
 * the caller use the camera frame timestamps it already has.
 *
 * <p>Methods are synchronised because frames are fed in from the camera analysis thread
 * while the UI reads progress and resets from the main thread.
 */
public final class HoldTimer {

    /** How long alignment must be held before the gate passes. */
    public static final long DEFAULT_HOLD_MS = 3000L;

    /** How long a dropout is forgiven before the count restarts. */
    public static final long DEFAULT_GRACE_MS = 300L;

    private final long holdMs;
    private final long graceMs;

    /** When the current unbroken hold began, or -1 when not counting. */
    private long holdStartMs = -1L;
    /** Timestamp of the most recent in-tolerance frame, used to size the grace window. */
    private long lastGoodMs = -1L;
    private boolean satisfied = false;

    public HoldTimer() {
        this(DEFAULT_HOLD_MS, DEFAULT_GRACE_MS);
    }

    public HoldTimer(long holdMs, long graceMs) {
        this.holdMs = holdMs;
        this.graceMs = graceMs;
    }

    /**
     * Feeds one frame's verdict in.
     *
     * @param inTolerance whether this frame met every alignment tolerance
     * @param nowMs       monotonic timestamp in milliseconds
     * @return hold progress from 0 to 1; 1 means the dwell is complete
     */
    public synchronized float update(boolean inTolerance, long nowMs) {
        if (inTolerance) {
            if (holdStartMs < 0) holdStartMs = nowMs;
            lastGoodMs = nowMs;
        } else if (holdStartMs >= 0 && nowMs - lastGoodMs > graceMs) {
            reset();
            return 0f;
        }

        if (holdStartMs < 0) return 0f;

        // Progress is measured to the last good frame, not to now: time spent inside the
        // grace window is forgiven, not credited.
        long held = lastGoodMs - holdStartMs;
        if (held >= holdMs) {
            satisfied = true;
            return 1f;
        }
        return Math.max(0f, Math.min(1f, held / (float) holdMs));
    }

    /** True once a full dwell has completed, and until {@link #reset()} is called. */
    public synchronized boolean isSatisfied() {
        return satisfied;
    }

    /** Drops the current count. Call after acting on a satisfied timer. */
    public synchronized void reset() {
        holdStartMs = -1L;
        lastGoodMs = -1L;
        satisfied = false;
    }

    /** Milliseconds still to go, rounded up; 0 once satisfied. */
    public synchronized long remainingMs() {
        if (satisfied) return 0L;
        if (holdStartMs < 0) return holdMs;
        return Math.max(0L, holdMs - (lastGoodMs - holdStartMs));
    }
}
