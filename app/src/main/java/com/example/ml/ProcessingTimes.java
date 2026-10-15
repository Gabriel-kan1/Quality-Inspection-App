package com.example.ml;

import android.os.Build;
import android.os.SystemClock;

import java.util.Locale;

/**
 * Wall-clock time of each inspection stage on this device, for the processing-time readout on
 * the results screen.
 *
 * <p>Held in memory only: nothing here is written to the database, the files directory or
 * anywhere else, and the app has no network access, so a number leaves the device only if the
 * operator copies it out. That is the point — these figures replace estimates in the report,
 * and the person running the test decides what to share.
 *
 * <p>Times are {@link SystemClock#elapsedRealtimeNanos()} differences, reported in
 * milliseconds. Stages are the ones the report's timing table lists; anything between them
 * (database writes, the noise gate, choosing what to ask about) is the gap between their sum
 * and {@link #afterCaptureMs}.
 */
public final class ProcessingTimes {

    /** Decoding the photo to the working size, opening the source for crops, OpenCV copy. */
    public long loadMs = -1;
    /** Looking for the calibration marker. */
    public long markerMs = -1;
    /** The edge stage: every method it compared, or the one it ran. */
    public long edgeMs = -1;
    /** How many edge methods that stage ran (more than one only when scoring against a layout). */
    public int edgeMethods;
    /** Contours to candidate boxes. */
    public long boxesMs = -1;
    /** Cutting every crop from the source photo. */
    public long cropMs;
    /** Encoding every crop and classifying it against the library. */
    public long embedMs;
    /** Writing every crop to the learning store. */
    public long saveMs;
    /** Crops encoded; {@link #embedMs} / this is the per-crop figure. */
    public int crops;
    /** Loading the encoder model, when this run was the one that loaded it, else 0. */
    public long encoderLoadMs;
    /** The whole detection call, start to finish. */
    public long detectionMs = -1;
    /** Registering the detections against the reference layout. */
    public long localiseMs = -1;
    /**
     * From the start of analysis until the background work is done — before any review
     * question is shown, so the operator's thinking time is never counted. The readout's
     * total adds {@link #localiseMs}, which runs afterwards on the main thread.
     */
    public long afterCaptureMs = -1;

    public static long now() {
        return SystemClock.elapsedRealtimeNanos();
    }

    public static long msSince(long startNanos) {
        return (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000L;
    }

    // ---- the live alignment gate, which runs on camera frames before the photo exists

    private static long gateNanos;
    private static int gateFrames;
    private static long gateWorstNanos;

    /** Starts a new capture session's gate statistics. */
    public static synchronized void resetGate() {
        gateNanos = 0;
        gateFrames = 0;
        gateWorstNanos = 0;
    }

    /** Records one camera frame's gate evaluation. */
    public static synchronized void recordGateFrame(long nanos) {
        gateNanos += nanos;
        gateFrames++;
        gateWorstNanos = Math.max(gateWorstNanos, nanos);
    }

    /** The readout, one stage per line, ready to copy. */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US, "Device: %s %s, Android %s, %d cores%n",
                Build.MANUFACTURER, Build.MODEL, Build.VERSION.RELEASE,
                Runtime.getRuntime().availableProcessors()));
        line(sb, "Load photo and working image", loadMs);
        line(sb, "Calibration marker search", markerMs);
        line(sb, edgeMethods > 1
                ? String.format(Locale.US, "Edge stage (%d methods compared)", edgeMethods)
                : "Edge stage (one method)", edgeMs);
        line(sb, "Candidate boxes", boxesMs);
        line(sb, String.format(Locale.US, "Crops cut (%d)", crops), cropMs);
        line(sb, String.format(Locale.US, "Embedding and classifying (%d crops)", crops), embedMs);
        if (crops > 0) {
            sb.append(String.format(Locale.US, "  per crop: %.1f ms%n", embedMs / (double) crops));
        }
        line(sb, "Saving crops", saveMs);
        if (encoderLoadMs > 0) line(sb, "Encoder model load (first run only)", encoderLoadMs);
        line(sb, "Detection total", detectionMs);
        line(sb, "Localisation", localiseMs);
        line(sb, "Total after capture (not counting review questions)",
                afterCaptureMs < 0 ? -1 : afterCaptureMs + Math.max(localiseMs, 0));
        synchronized (ProcessingTimes.class) {
            if (gateFrames > 0) {
                sb.append(String.format(Locale.US,
                        "Alignment gate per frame: mean %.0f ms, worst %d ms (%d frames)%n",
                        gateNanos / 1e6 / gateFrames, gateWorstNanos / 1_000_000L, gateFrames));
            } else {
                sb.append("Alignment gate per frame: not measured (no live camera frames)\n");
            }
        }
        return sb.toString().trim();
    }

    private static void line(StringBuilder sb, String label, long ms) {
        sb.append(label).append(": ").append(ms < 0 ? "n/a" : ms + " ms").append('\n');
    }
}
