package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.core.Mat;

import java.util.Locale;

/**
 * Decides whether a live camera frame is framed closely enough like the part's reference
 * photo to be worth inspecting, and tells the operator how to move if it is not.
 *
 * <h3>Two questions, two methods</h3>
 *
 * <b>"Is this the right area?"</b> is answered by the DINOv2 encoder already shipped for
 * feature classification — a cosine similarity between the frame and the reference. It
 * describes what the area <em>is</em> rather than which corners it contains, which matters
 * because the hardware is exactly what changes between an inspection and its reference.
 * ORB keypoints plus a RANSAC homography was tried and measured first and carries no signal
 * here: on {@code screws.jpg} against {@code screws_test.jpg} — the same four-hole cluster —
 * 8 of ~1000 keypoints survived the ratio test, against 2 for a completely different part of
 * the chassis.
 *
 * <b>"How should the camera move?"</b> is answered by {@link TemplateMatcher}, which locates
 * the frame inside the reference (or the reference inside the frame) and returns a framing
 * ratio and an aim offset. An embedding cannot do this: it is a single number with no
 * spatial meaning, so it can say "wrong" but never "40% too close and aimed left".
 *
 * <p>The split is not a preference, it is what the measurements support. The template score
 * does <b>not</b> separate the reference area from a different one — same-area scores ran
 * down to 0.285 while different-area scores reached 0.456 on the same reference — so it must
 * not be promoted to the area test. Equally, the encoder's own scale probe was tried as a
 * framing test and could not report direction at all. See {@link TemplateMatcher}'s notes for
 * the representation study behind the matching, and re-run both before changing either half.
 *
 * <h3>Known limits</h3>
 *
 * The match models translation and uniform scale only, so advice degrades as the camera
 * <em>angle</em> changes rather than its distance; roll is not modelled at all. And the area
 * test is only as good as the reference photo: a wide reference whose subject occupies a few
 * percent of the frame scores unrelated close-ups as high as its own (measured on
 * {@code screws_flash.jpg}, where four different-area shots outranked the reference's own
 * cluster). A reference-quality check at setup time is the fix for that, and does not exist.
 *
 * <p>Instances are not thread-safe; {@link #evaluate} is meant to be called from a single
 * camera-analysis thread.
 */
public final class AlignmentGate {

    private static final String TAG = "AlignmentGate";

    /**
     * Cosine similarity a frame must reach to count as the reference area.
     *
     * <p>Fitted over 252 labelled asset pairs in {@code python/scale_probe_study.py}: 0.70 is
     * the knee, and the last value that costs no true accepts. 0.65 gives 9 false accepts and
     * 0 false rejects, 0.70 gives 5 and 0, 0.74 gives 2 and 5, 0.80 gives 2 and 9. Recall
     * against the RANSAC-verified pairs from {@code python/pair_ground_truth.py} falls from
     * 17/19 to 15/19 over that same move.
     */
    private static final float MATCH_THRESHOLD = 0.7f;

    /** Below this fraction of the threshold the frame shows nothing relevant at all, which
     *  earns a blunter hint than "keep adjusting". */
    private static final float NO_MATCH_FRACTION = 0.7f;

    /**
     * Framing ratio band the frame must sit in, as a multiple of the reference's field of
     * view. Below 1 the frame covers less of the part than the reference did, so expected
     * hardware falls outside it — that is the direction that produces a confident report
     * about features nobody photographed, so the floor is the tighter of the two.
     *
     * <p>Provisional. The matcher recovers this ratio to within 0.02 when only lighting
     * differs, and considerably worse when the camera angle does; these are set from that
     * spread rather than from a threshold sweep, which needs frames the asset set does not
     * contain. Re-fit once real captures exist.
     */
    private static final float SCALE_MIN = 0.85f;
    private static final float SCALE_MAX = 1.35f;

    /** How far the aim may be off, as a fraction of the frame. Offset error measured 0.037
     *  under lighting change, so this leaves room without letting the part drift off-frame. */
    private static final float OFFSET_TOLERANCE = 0.12f;

    /** Below this the correlation peak is noise and no advice can be trusted. */
    private static final float MIN_TEMPLATE_SCORE = 0.20f;

    /**
     * Mean per-pixel luminance change between consecutive thumbnails, above which the phone
     * counts as moving. 0-255 scale, per {@link #NOMINAL_INTERVAL_MS}. Camera noise alone
     * runs about 1-2 on a still handheld phone. Unlike the constants above, this one has no
     * study behind it.
     */
    private static final float MAX_MOTION = 6.0f;

    /**
     * The interval {@link #MAX_MOTION} is expressed at; raw deltas are scaled to it so the
     * limit is a rate. Analysed frames arrive at whatever rate {@link #evaluate} runs at, so
     * an unscaled delta made the same hand speed read differently depending on how much work
     * the frame happened to cost. 100 ms is an assumption, not a measurement;
     * {@link Reading#debugLine()} reports the observed interval so it can be checked on real
     * hardware.
     */
    private static final long NOMINAL_INTERVAL_MS = 100L;

    /** Below this, per-frame sensor noise dominates once divided out. */
    private static final long MIN_INTERVAL_MS = 50L;

    /** Beyond this the frames are not consecutive, so the pair is discarded and re-based. */
    private static final long STALE_INTERVAL_MS = 500L;

    /** Edge of the square thumbnail used for the motion estimate. */
    private static final int MOTION_THUMB = 32;

    /** Why a frame was rejected, in the order the operator should act on it. */
    public enum Fault {
        NONE, NO_MATCH, WEAK_MATCH, NO_FRAMING, TOO_CLOSE, TOO_FAR,
        AIM_LEFT, AIM_RIGHT, AIM_UP, AIM_DOWN, MOVING
    }

    /** One frame's verdict. Immutable so it can be handed straight to the UI thread. */
    public static final class Reading {
        /** True when the frame may count toward the three-second hold. */
        public final boolean inTolerance;
        public final Fault fault;
        /** Short imperative instruction for the operator, with the magnitude of the error. */
        public final String hint;
        /** Cosine similarity to the reference, -1..1. NaN when the frame could not be encoded. */
        public final float similarity;
        /** Template match peak, -1..1. NaN when the framing search could not run. */
        public final float score;
        /** Frame field of view as a multiple of the reference's. NaN when unknown. */
        public final float scale;
        /** Aim error as a fraction of the frame; positive means pan right. NaN when unknown. */
        public final float offsetX;
        /** Positive means tilt down. */
        public final float offsetY;
        /** Mean luminance change since the previous frame, per {@link #NOMINAL_INTERVAL_MS}. */
        public final float motion;
        /** What {@link #motion} was measured over, or -1. Debug only. */
        public final long motionIntervalMs;

        Reading(boolean inTolerance, Fault fault, String hint, float similarity, float score,
                float scale, float offsetX, float offsetY, float motion, long motionIntervalMs) {
            this.inTolerance = inTolerance;
            this.fault = fault;
            this.hint = hint;
            this.similarity = similarity;
            this.score = score;
            this.scale = scale;
            this.offsetX = offsetX;
            this.offsetY = offsetY;
            this.motion = motion;
            this.motionIntervalMs = motionIntervalMs;
        }

        /** Area-match strength as a 0-100 figure for display. */
        public int matchPercent() {
            if (Float.isNaN(similarity)) return 0;
            return Math.max(0, Math.min(100, Math.round(similarity * 100f)));
        }

        /** Template correlation as a 0-100 figure. Its useful range is compressed — a good
         *  match sits near 45 — so it reads as a confidence, not as a percentage correct. */
        public int templatePercent() {
            if (Float.isNaN(score)) return 0;
            return Math.max(0, Math.min(100, Math.round(score * 100f)));
        }

        /** Compact numeric dump for the debug readout and logcat. */
        public String debugLine() {
            StringBuilder sb = new StringBuilder(String.format(Locale.US,
                    "sim %.3f (need %.2f)", similarity, MATCH_THRESHOLD));
            if (!Float.isNaN(score)) {
                sb.append(String.format(Locale.US, " · tm %.3f scale %.2f (%.2f-%.2f)",
                        score, scale, SCALE_MIN, SCALE_MAX));
                sb.append(String.format(Locale.US, " aim %+.2f/%+.2f (max %.2f)",
                        offsetX, offsetY, OFFSET_TOLERANCE));
            }
            sb.append(String.format(Locale.US, " · motion %.1f", motion));
            if (motionIntervalMs >= 0) sb.append(String.format(Locale.US, " over %dms", motionIntervalMs));
            sb.append(String.format(Locale.US, " (max %.1f per %dms)", MAX_MOTION, NOMINAL_INTERVAL_MS));
            return sb.toString();
        }
    }

    private final Encoder embedder;
    private final float[] referenceEmbedding;
    /** Gradient image of the reference, prepared once. Null disables framing advice. */
    private final Mat referenceEdges;

    private int[] previousThumb;
    private long previousThumbMs;

    AlignmentGate(Encoder embedder, float[] referenceEmbedding, Mat referenceEdges) {
        this.embedder = embedder;
        this.referenceEmbedding = referenceEmbedding;
        this.referenceEdges = referenceEdges;
    }

    /**
     * Builds a gate from the part's reference photo.
     *
     * @return null when the encoder is unavailable or the reference could not be encoded, in
     *         which case the caller must block inspection rather than pass everything.
     */
    public static AlignmentGate fromReference(Context context, Bitmap reference) {
        if (reference == null) return null;
        FeatureEmbedder embedder = FeatureEmbedder.get(context);
        if (embedder == null) {
            Log.e(TAG, "Feature encoder unavailable — cannot gate on alignment.");
            return null;
        }
        float[] vector = embedder.embed(reference);
        if (vector == null) {
            Log.e(TAG, "Reference image could not be encoded.");
            return null;
        }
        normalise(vector);

        Mat edges = TemplateMatcher.prepare(reference);
        if (edges == null) {
            Log.w(TAG, "Reference gradients unavailable — framing advice is disabled.");
        }
        return new AlignmentGate(embedder, vector, edges);
    }

    /**
     * Scores one frame. The frame must already be rotated to upright display orientation:
     * neither the encoder nor the template search is rotation invariant, and the reference
     * was prepared upright.
     *
     * @param nowMs monotonic timestamp, taken before the call so the motion rate excludes the
     *              work this method is about to do.
     */
    public Reading evaluate(Bitmap frame, long nowMs) {
        if (frame == null) {
            return reading(false, Fault.NO_MATCH, "NO FRAME", Float.NaN, null, Float.NaN, -1L);
        }

        long interval = previousThumb != null ? nowMs - previousThumbMs : -1L;
        float motion = measureMotion(frame, nowMs);

        float[] vector = embedder.embed(frame);
        if (vector == null) {
            return reading(false, Fault.NO_MATCH, "NO MATCH — POINT AT THE PART",
                    Float.NaN, null, motion, interval);
        }
        normalise(vector);
        float similarity = dot(vector, referenceEmbedding);

        // The framing search costs more than the encoder pass, and a frame showing the wrong
        // area gets the same hint whatever it says, so it is only worth paying for once the
        // area is plausible.
        TemplateMatcher.Match match = null;
        if (similarity >= MATCH_THRESHOLD && referenceEdges != null) {
            Mat frameEdges = TemplateMatcher.prepare(frame);
            if (frameEdges != null) {
                try {
                    match = TemplateMatcher.match(referenceEdges, frameEdges);
                } finally {
                    frameEdges.release();
                }
            }
        }
        return classify(similarity, match, motion, interval);
    }

    /** Convenience for tests with no framing search. */
    static Reading classify(float similarity, float motion) {
        return classify(similarity, null, motion, -1L);
    }

    /**
     * Applies the thresholds and picks the single most useful thing to tell the operator.
     *
     * <p>Order is area, then framing distance, then aim, then steadiness. Distance outranks
     * aim because it decides whether the part is even fully in shot, and both outrank motion:
     * a phone held wrong is steady-wrong, and "hold still" would strand the operator there.
     */
    static Reading classify(float similarity, TemplateMatcher.Match match, float motion,
                            long intervalMs) {
        if (similarity < MATCH_THRESHOLD) {
            // The area score belongs on the area faults and nowhere else. It is what the
            // operator is climbing toward the threshold here, and it is meaningless beside a
            // framing rejection — printing it there read as "78% match, rejected".
            int pct = Math.max(0, Math.min(100, Math.round(similarity * 100f)));
            if (similarity < MATCH_THRESHOLD * NO_MATCH_FRACTION) {
                return reading(false, Fault.NO_MATCH, String.format(Locale.US,
                                "NO MATCH — POINT AT THE PART (area %d%%)", pct),
                        similarity, match, motion, intervalMs);
            }
            return reading(false, Fault.WEAK_MATCH, String.format(Locale.US,
                            "CLOSE — FRAME IT LIKE THE GHOST (area %d%%)", pct),
                    similarity, match, motion, intervalMs);
        }

        // The area matched but the framing could not be measured, so there is nothing to
        // certify. Passing here would let exactly the misframed capture through that the
        // search exists to catch.
        if (match == null || Float.isNaN(match.scale) || match.score < MIN_TEMPLATE_SCORE) {
            return reading(false, Fault.NO_FRAMING, "CANNOT READ FRAMING — STEADY AND RETRY",
                    similarity, match, motion, intervalMs);
        }

        if (match.scale < SCALE_MIN) {
            return reading(false, Fault.TOO_CLOSE, String.format(Locale.US,
                            "TOO CLOSE — MOVE BACK (showing %.0f%% of the reference)",
                            match.scale * 100f),
                    similarity, match, motion, intervalMs);
        }
        if (match.scale > SCALE_MAX) {
            return reading(false, Fault.TOO_FAR, String.format(Locale.US,
                            "TOO FAR — MOVE CLOSER (showing %.1fx the reference)", match.scale),
                    similarity, match, motion, intervalMs);
        }

        if (match.worstOffset() > OFFSET_TOLERANCE) {
            boolean horizontal = Math.abs(match.offsetX) >= Math.abs(match.offsetY);
            float amount = (horizontal ? Math.abs(match.offsetX) : Math.abs(match.offsetY)) * 100f;
            Fault fault;
            String direction;
            if (horizontal) {
                fault = match.offsetX > 0 ? Fault.AIM_RIGHT : Fault.AIM_LEFT;
                direction = match.offsetX > 0 ? "RIGHT" : "LEFT";
            } else {
                fault = match.offsetY > 0 ? Fault.AIM_DOWN : Fault.AIM_UP;
                direction = match.offsetY > 0 ? "DOWN" : "UP";
            }
            return reading(false, fault, String.format(Locale.US,
                            "AIM %s (%.0f%% off centre)", direction, amount),
                    similarity, match, motion, intervalMs);
        }

        // The first frame has no predecessor, so it cannot be judged moving; the hold needs
        // several frames anyway, so nothing is let through early.
        if (!(Float.isNaN(motion) || motion <= MAX_MOTION)) {
            return reading(false, Fault.MOVING, "STEADY THE PHONE",
                    similarity, match, motion, intervalMs);
        }
        return reading(true, Fault.NONE, "HOLD STILL", similarity, match, motion, intervalMs);
    }

    private static Reading reading(boolean ok, Fault fault, String hint, float similarity,
                                   TemplateMatcher.Match m, float motion, long intervalMs) {
        return new Reading(ok, fault, hint, similarity,
                m == null ? Float.NaN : m.score,
                m == null ? Float.NaN : m.scale,
                m == null ? Float.NaN : m.offsetX,
                m == null ? Float.NaN : m.offsetY,
                motion, intervalMs);
    }

    /**
     * Mean absolute luminance change against the previous frame on a 0-255 scale, scaled to
     * {@link #NOMINAL_INTERVAL_MS}. NaN when there is no predecessor, or none close enough in
     * time to give a rate.
     */
    float measureMotion(Bitmap frame, long nowMs) {
        Bitmap thumb = Bitmap.createScaledBitmap(frame, MOTION_THUMB, MOTION_THUMB, true);
        int[] pixels = new int[MOTION_THUMB * MOTION_THUMB];
        thumb.getPixels(pixels, 0, MOTION_THUMB, 0, 0, MOTION_THUMB, MOTION_THUMB);
        if (thumb != frame) thumb.recycle();

        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            pixels[i] = (((p >> 16) & 0xFF) * 77 + ((p >> 8) & 0xFF) * 151 + (p & 0xFF) * 28) >> 8;
        }

        float motion = Float.NaN;
        if (previousThumb != null) {
            long elapsed = nowMs - previousThumbMs;
            if (elapsed >= MIN_INTERVAL_MS && elapsed <= STALE_INTERVAL_MS) {
                long total = 0;
                for (int i = 0; i < pixels.length; i++) {
                    total += Math.abs(pixels[i] - previousThumb[i]);
                }
                motion = (float) total / pixels.length * NOMINAL_INTERVAL_MS / elapsed;
            }
        }
        previousThumb = pixels;
        previousThumbMs = nowMs;
        return motion;
    }

    /** Forgets the motion history, so the next frame is not compared against a stale one. */
    public void resetMotion() {
        previousThumb = null;
        previousThumbMs = 0L;
    }

    private static void normalise(float[] v) {
        double sum = 0;
        for (float x : v) sum += x * x;
        double norm = Math.sqrt(sum);
        if (norm <= 0) return;
        for (int i = 0; i < v.length; i++) v[i] /= (float) norm;
    }

    private static float dot(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        float sum = 0;
        for (int i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    /** Releases the reference gradients; the encoder is a shared singleton owned elsewhere. */
    public void release() {
        resetMotion();
        if (referenceEdges != null) referenceEdges.release();
    }
}
