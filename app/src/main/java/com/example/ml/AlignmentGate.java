package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.util.Locale;

/**
 * Decides whether a live camera frame is showing the same area of the chassis as the
 * part's reference photo, closely enough framed to be worth inspecting.
 *
 * <h3>Why this is an embedding comparison and not feature matching</h3>
 *
 * The obvious approach — ORB keypoints plus a RANSAC homography — was tried and measured
 * against the project's own assets, and it does not work here. On {@code screws.jpg} the
 * chassis plate is a near-featureless matte surface, so almost every keypoint lands on the
 * hardware itself. That hardware is exactly what changes between an inspection and its
 * reference: a missing stud or a swapped screw removes the very features the match depends
 * on. Scoring {@code screws.jpg} against {@code screws_test.jpg} (the same four-hole
 * cluster, different fasteners, one hole empty) survived 8 of ~1000 keypoints through the
 * ratio test — statistically indistinguishable from {@code corner.jpg}, a completely
 * different part of the chassis, which survived 2. The signal was absent, not merely weak,
 * so no amount of threshold loosening would have recovered it.
 *
 * <p>Detected-feature geometry was tried next — matching the spatial pattern of hardware
 * positions rather than their appearance — and failed for a related reason: the detector
 * merges the four fasteners in {@code screws.jpg} into a single box, and a different area
 * ({@code defected.jpg}) scored closer to the reference than any genuine match did.
 *
 * <p>The DINOv2 encoder already shipped for feature classification does carry the signal,
 * because it describes what the area <em>is</em> rather than which corners it contains.
 * Measured over the bundled assets with {@code screws.jpg} as the reference:
 *
 * <pre>
 *   same area:      screws_test 0.747   reference 0.739   defected 0.732
 *                   screws_three 0.659  screws_flash 0.603  screws_two 0.578
 *   different area: zero 0.510   one 0.368   three 0.358   corner 0.309
 *                   whole 0.218  whole_light 0.198
 * </pre>
 *
 * <h3>What this costs</h3>
 *
 * An embedding says "right area" but not "off by 12% to the left", so there are no
 * directional pan or rotate hints — that geometry genuinely is not recoverable from these
 * images by any method tested. Framing is still gated, just indirectly: similarity falls
 * off as the operator backs away, which is why the extreme wide shot {@code screws_none}
 * (0.499) lands below threshold. What the operator gets instead is a match strength to
 * maximise, plus a steadiness check so the captured frame is not motion-blurred.
 *
 * <p>Instances are not thread-safe; {@link #evaluate} is meant to be called from a single
 * camera-analysis thread.
 */
public final class AlignmentGate {

    private static final String TAG = "AlignmentGate";

    // ─────────────────────────────────────────────────────────────
    // CONFIG — non-final so they can be adjusted at runtime while tuning against a real
    // chassis without recompiling the logic.
    // ─────────────────────────────────────────────────────────────

    /**
     * Cosine similarity a frame must reach to count as the reference area.
     *
     * <p>0.57 sits in the gap measured over the bundled assets: it clears every same-area
     * shot that is close enough to inspect (lowest is screws_two at 0.578) and rejects
     * every different-area shot (highest is zero.jpg at 0.510). The margin is ~0.07, so
     * expect to re-fit this against your own reference photos — the debug readout on the
     * inspection screen shows the live figure.
     */
    private static float MATCH_THRESHOLD = 0.57f;

    /** Below this fraction of the threshold the frame is treated as showing nothing
     *  relevant at all, which earns a blunter hint than "keep adjusting". */
    private static float NO_MATCH_FRACTION = 0.6f;

    /**
     * Mean per-pixel luminance change between consecutive thumbnails, above which the phone
     * counts as moving. 0-255 scale. Camera noise alone runs about 1-2 on a still handheld
     * phone, so this sits far enough above the noise floor not to trip on it.
     */
    private static float MAX_MOTION = 6.0f;

    /** Edge of the square thumbnail used for the motion estimate. Small on purpose: this
     *  measures gross camera movement, not detail. */
    private static final int MOTION_THUMB = 32;

    // ─────────────────────────────────────────────────────────────

    /** Why a frame was rejected, in the order the operator should act on it. */
    public enum Fault { NONE, NO_MATCH, WEAK_MATCH, MOVING }

    /** One frame's verdict. Immutable so it can be handed straight to the UI thread. */
    public static final class Reading {
        /** True when the frame may count toward the three-second hold. */
        public final boolean inTolerance;
        public final Fault fault;
        /** Short, imperative instruction for the operator. */
        public final String hint;
        /** Cosine similarity to the reference, -1..1. NaN when the frame could not be encoded. */
        public final float similarity;
        /** Mean luminance change since the previous frame, 0-255. NaN on the first frame. */
        public final float motion;

        Reading(boolean inTolerance, Fault fault, String hint, float similarity, float motion) {
            this.inTolerance = inTolerance;
            this.fault = fault;
            this.hint = hint;
            this.similarity = similarity;
            this.motion = motion;
        }

        /** Match strength as a 0-100 figure for display. */
        public int matchPercent() {
            if (Float.isNaN(similarity)) return 0;
            return Math.max(0, Math.min(100, Math.round(similarity * 100f)));
        }

        /** Compact numeric dump for the debug readout and logcat. */
        public String debugLine() {
            return String.format(Locale.US, "sim %.3f (need %.2f) · motion %.1f (max %.1f)",
                    similarity, MATCH_THRESHOLD, motion, MAX_MOTION);
        }
    }

    private final FeatureEmbedder embedder;
    private final float[] referenceEmbedding;

    /** Luminance thumbnail of the previous frame, for the motion estimate. */
    private int[] previousThumb;

    private AlignmentGate(FeatureEmbedder embedder, float[] referenceEmbedding) {
        this.embedder = embedder;
        this.referenceEmbedding = referenceEmbedding;
    }

    /**
     * Builds a gate from the part's reference photo.
     *
     * @return null when the encoder is unavailable or the reference could not be encoded,
     *         in which case the caller must block inspection rather than pass everything.
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
        Log.d(TAG, "Reference encoded, " + vector.length + " dims.");
        return new AlignmentGate(embedder, vector);
    }

    /**
     * Scores one frame. The frame must already be rotated to upright display orientation:
     * the encoder is not rotation invariant, and the reference was encoded upright.
     *
     * @return the verdict; never null.
     */
    public Reading evaluate(Bitmap frame) {
        if (frame == null) {
            return new Reading(false, Fault.NO_MATCH, "NO FRAME", Float.NaN, Float.NaN);
        }

        float motion = measureMotion(frame);

        float[] vector = embedder.embed(frame);
        if (vector == null) {
            return new Reading(false, Fault.NO_MATCH, "NO MATCH — POINT AT THE PART",
                    Float.NaN, motion);
        }
        normalise(vector);
        float similarity = dot(vector, referenceEmbedding);

        return classify(similarity, motion);
    }

    /** Applies the thresholds and picks the single most useful thing to tell the operator. */
    static Reading classify(float similarity, float motion) {
        boolean matches = similarity >= MATCH_THRESHOLD;
        // The first frame has no predecessor to compare against, so it cannot be judged
        // moving; the hold needs several frames anyway, so nothing is let through early.
        boolean steady = Float.isNaN(motion) || motion <= MAX_MOTION;

        if (matches && steady) {
            return new Reading(true, Fault.NONE, "HOLD STILL", similarity, motion);
        }
        if (matches) {
            return new Reading(false, Fault.MOVING, "STEADY THE PHONE", similarity, motion);
        }
        if (similarity < MATCH_THRESHOLD * NO_MATCH_FRACTION) {
            return new Reading(false, Fault.NO_MATCH, "NO MATCH — POINT AT THE PART",
                    similarity, motion);
        }
        // Close but short: the operator is looking at roughly the right thing, and moving
        // closer or squaring up is what raises the score.
        return new Reading(false, Fault.WEAK_MATCH, "CLOSE — FRAME IT LIKE THE GHOST",
                similarity, motion);
    }

    /**
     * Mean absolute luminance change against the previous frame, on a 0-255 scale.
     * Returns NaN for the first frame.
     */
    private float measureMotion(Bitmap frame) {
        Bitmap thumb = Bitmap.createScaledBitmap(frame, MOTION_THUMB, MOTION_THUMB, true);
        int[] pixels = new int[MOTION_THUMB * MOTION_THUMB];
        thumb.getPixels(pixels, 0, MOTION_THUMB, 0, 0, MOTION_THUMB, MOTION_THUMB);
        if (thumb != frame) thumb.recycle();

        // Overwrite the colour thumbnail with luminance in place; only intensity matters.
        for (int i = 0; i < pixels.length; i++) {
            int p = pixels[i];
            pixels[i] = (((p >> 16) & 0xFF) * 77 + ((p >> 8) & 0xFF) * 151 + (p & 0xFF) * 28) >> 8;
        }

        float motion = Float.NaN;
        if (previousThumb != null) {
            long total = 0;
            for (int i = 0; i < pixels.length; i++) {
                total += Math.abs(pixels[i] - previousThumb[i]);
            }
            motion = (float) total / pixels.length;
        }
        previousThumb = pixels;
        return motion;
    }

    /** Forgets the motion history, so the next frame is not compared against a stale one. */
    public void resetMotion() {
        previousThumb = null;
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

    /** Releases nothing native — the encoder is a shared singleton owned by FeatureEmbedder. */
    public void release() {
        previousThumb = null;
    }
}
