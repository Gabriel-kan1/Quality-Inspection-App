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
 * <h3>Why similarity alone does not settle shooting distance</h3>
 *
 * The encoder is partly scale tolerant, so "the right area photographed from twice as far
 * away" still scores well: against {@code stud1.jpg} the mid-range {@code screws_two.jpg}
 * reaches 0.717 and {@code screws_three.jpg} 0.676, both clearing the threshold despite
 * framing the cluster at a fraction of the reference's size. Raising the threshold is not
 * the fix — it takes genuine same-distance shots with it, and it cannot tell the operator
 * which way to move.
 *
 * <p>What separates the two cases is a <em>scale probe</em>: re-score the frame's centre
 * 50% crop against the reference. If zooming in improves the match, the frame was showing
 * the area too small — the operator is too far away. The mirror image, scoring the frame
 * against a centre crop of the <em>reference</em>, catches being too close. Measured over
 * 252 labelled asset pairs (python/scale_probe_study.py):
 *
 * <pre>
 *   crop-in gain      correctly framed  mean -0.070 (p90 +0.012)
 *                     too far away      mean +0.156 (p10 +0.025)
 *   crop-reference    correctly framed  mean -0.087 (p90 +0.008)
 *   gain              too close         mean +0.218 (p10 +0.159)
 * </pre>
 *
 * The populations barely overlap, so {@link #SCALE_GAIN_TOLERANCE} sits between them. At
 * the current threshold this removes two thirds of the wrong-distance false accepts (15 to
 * 5 over that pair set) while rejecting none of the correctly framed shots — measured at
 * every threshold from 0.60 to 0.80, the probe never cost a true accept: the false-reject
 * column is identical with the probe on and off. Two cues that sound more direct were
 * tried first and both failed: median
 * detected-blob diameter wandered by 3x across crops of one image (the contours are
 * surface grain, not hardware), and the texture's spatial-frequency centroid flattened
 * out at close range where sensor noise dominates the spectrum, and moved the wrong way
 * entirely on {@code corner.jpg}.
 *
 * <h3>What this costs</h3>
 *
 * An embedding says "right area" but not "off by 12% to the left", so there are no
 * directional pan or rotate hints — that geometry genuinely is not recoverable from these
 * images by any method tested. Distance is now reported ("move closer" / "back off"),
 * bearing still is not. The probe costs one extra encoder pass, charged only on frames
 * that already clear the threshold: below it the frame is rejected regardless, so the
 * probe would not change the verdict. The too-close side is free — the cropped reference
 * is encoded once when the gate is built.
 *
 * <p>Instances are not thread-safe; {@link #evaluate} is meant to be called from a single
 * camera-analysis thread.
 */
public final class AlignmentGate {

    private static final String TAG = "AlignmentGate";

    // ─────────────────────────────────────────────────────────────
    // CONFIG — these were left non-final "so they can be adjusted at runtime while tuning",
    // but nothing ever adjusted them: there is no setter and no debug screen, so they were
    // constants that had merely lost the keyword. Restored to final. If runtime tuning is
    // wanted, add the surface that provides it rather than removing the keyword again.
    // ─────────────────────────────────────────────────────────────

    /**
     * Cosine similarity a frame must reach to count as the reference area.
     *
     * <p>Fitted to the gap measured over the bundled assets. With a close reference such as
     * {@code stud1.jpg}, same-area shots at the same distance sit at 0.81 and above while
     * different areas top out around 0.49; the widest reference set pushes the worst
     * different-area confusion to 0.72 ({@code head1.jpg} against {@code one.jpg}, two
     * extreme close-ups of near-identical plain metal), which no threshold in this range
     * separates. Expect to re-fit against your own reference photos — the debug readout on
     * the inspection screen shows the live figure. Note that shooting distance is no longer
     * this threshold's job; see {@link #SCALE_GAIN_TOLERANCE}.
     *
     * <p>0.70 is the knee of both studies, and is the last value that costs nothing. Over the
     * 252 labelled pairs in {@code python/scale_probe_study.py}, with the probe on:
     *
     * <pre>
     *   threshold   false accepts   false rejects   proven-pair recall
     *     0.65            9               0              19/19
     *     0.70            5               0              17/19
     *     0.74            2               5              15/19
     *     0.80            2               9              15/19
     * </pre>
     *
     * Recall is scored against the RANSAC-verified same-area pairs from
     * {@code python/pair_ground_truth.py}, which is the provenance for this number — re-run
     * both scripts before changing it. Tightening past 0.70 buys three fewer false accepts
     * and pays nine true accepts for them, including pairs proven same-area by several
     * hundred inliers ({@code screws_test}/{@code screws_two}, {@code screws2}/{@code
     * screws_test}). Loosening to 0.65 recovers the last two proven pairs but nearly doubles
     * the false accepts.
     */
    private static final float MATCH_THRESHOLD = 0.7f;

    /** Below this fraction of the threshold the frame is treated as showing nothing
     *  relevant at all, which earns a blunter hint than "keep adjusting". */
    private static final float NO_MATCH_FRACTION = 0.7f;

    /**
     * How much a scale probe may improve the match before the frame counts as being shot
     * from the wrong distance rather than merely framed loosely.
     *
     * <p>0.10 sits above the p90 of correctly framed frames (+0.012) and below the p10 of
     * too-far ones (+0.025) with room to spare, deliberately biased towards letting a
     * borderline frame through: a gain this large means the reference area is visibly
     * smaller (or larger) than it was when the reference was saved, not that framing is
     * a few percent off.
     */
    private static final float SCALE_GAIN_TOLERANCE = 0.10f;

    /**
     * Fraction of each side kept by the probe crop. 0.5 quarters the visible area, which
     * is a big enough scale step to move the similarity decisively when the distance is
     * wrong; smaller probes clipped the hardware out of frame on correctly framed shots
     * and larger ones (0.75) blurred the two populations back together.
     */
    private static final float PROBE_CROP = 0.5f;

    /**
     * Mean per-pixel luminance change between consecutive thumbnails, above which the phone
     * counts as moving. 0-255 scale. Camera noise alone runs about 1-2 on a still handheld
     * phone, so this sits far enough above the noise floor not to trip on it.
     */
    private static final float MAX_MOTION = 6.0f;

    /** Edge of the square thumbnail used for the motion estimate. Small on purpose: this
     *  measures gross camera movement, not detail. */
    private static final int MOTION_THUMB = 32;

    // ─────────────────────────────────────────────────────────────

    /** Why a frame was rejected, in the order the operator should act on it. */
    public enum Fault { NONE, NO_MATCH, WEAK_MATCH, TOO_FAR, TOO_CLOSE, MOVING }

    /** One frame's verdict. Immutable so it can be handed straight to the UI thread. */
    public static final class Reading {
        /** True when the frame may count toward the three-second hold. */
        public final boolean inTolerance;
        public final Fault fault;
        /** Short, imperative instruction for the operator. */
        public final String hint;
        /** Cosine similarity to the reference, -1..1. NaN when the frame could not be encoded. */
        public final float similarity;
        /**
         * How much scoring the frame's centre crop instead improved the match. Positive
         * means the reference area appears too small — the phone is too far away. NaN when
         * the probe did not run, which is every frame that fell short of the threshold.
         */
        public final float farGain;
        /** The mirror measure against a cropped reference; positive means too close. */
        public final float closeGain;
        /** Mean luminance change since the previous frame, 0-255. NaN on the first frame. */
        public final float motion;

        Reading(boolean inTolerance, Fault fault, String hint, float similarity,
                float farGain, float closeGain, float motion) {
            this.inTolerance = inTolerance;
            this.fault = fault;
            this.hint = hint;
            this.similarity = similarity;
            this.farGain = farGain;
            this.closeGain = closeGain;
            this.motion = motion;
        }

        /**
         * Similarity after the distance penalty, which is what the operator is shown.
         *
         * <p>Raw cosine overstates a wrong-distance frame — the same hardware at half the
         * size still reads as broadly the right area — so a frame that only scores well
         * because the probe found the content at another scale is docked by exactly that
         * much. Correctly framed frames have negative gains and are untouched.
         */
        public float adjustedSimilarity() {
            if (Float.isNaN(similarity)) return Float.NaN;
            float penalty = 0f;
            if (!Float.isNaN(farGain)) penalty = Math.max(penalty, farGain);
            if (!Float.isNaN(closeGain)) penalty = Math.max(penalty, closeGain);
            return similarity - penalty;
        }

        /** Match strength as a 0-100 figure for display. */
        public int matchPercent() {
            float value = adjustedSimilarity();
            if (Float.isNaN(value)) return 0;
            return Math.max(0, Math.min(100, Math.round(value * 100f)));
        }

        /** Compact numeric dump for the debug readout and logcat. */
        public String debugLine() {
            String scale = "";
            if (!Float.isNaN(farGain) || !Float.isNaN(closeGain)) {
                scale = String.format(Locale.US, " · zoom %+.3f/%+.3f (max %.2f)",
                        farGain, closeGain, SCALE_GAIN_TOLERANCE);
            }
            return String.format(Locale.US, "sim %.3f (need %.2f) · motion %.1f (max %.1f)%s",
                    similarity, MATCH_THRESHOLD, motion, MAX_MOTION, scale);
        }
    }

    private final Encoder embedder;
    private final float[] referenceEmbedding;
    /** The reference's centre crop, encoded once so the too-close test is free per frame.
     *  Null when it could not be encoded, which disables that test rather than the gate. */
    private final float[] referenceZoomEmbedding;

    /** Luminance thumbnail of the previous frame, for the motion estimate. */
    private int[] previousThumb;

    // Package-visible so tests can build a gate around a null embedder and exercise the
    // parts that do not need one — the motion estimate in particular, which is plain bitmap
    // arithmetic gating every capture but was previously unreachable without an ONNX session.
    AlignmentGate(Encoder embedder, float[] referenceEmbedding,
                          float[] referenceZoomEmbedding) {
        this.embedder = embedder;
        this.referenceEmbedding = referenceEmbedding;
        this.referenceZoomEmbedding = referenceZoomEmbedding;
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

        float[] zoomVector = null;
        Bitmap zoomed = centreCrop(reference, PROBE_CROP);
        if (zoomed != null) {
            zoomVector = embedder.embed(zoomed);
            if (zoomed != reference) zoomed.recycle();
            if (zoomVector != null) {
                normalise(zoomVector);
            } else {
                Log.w(TAG, "Reference centre crop could not be encoded — "
                        + "frames that are too close will not be caught.");
            }
        }

        Log.d(TAG, "Reference encoded, " + vector.length + " dims.");
        return new AlignmentGate(embedder, vector, zoomVector);
    }

    /**
     * Scores one frame. The frame must already be rotated to upright display orientation:
     * the encoder is not rotation invariant, and the reference was encoded upright.
     *
     * @return the verdict; never null.
     */
    public Reading evaluate(Bitmap frame) {
        if (frame == null) {
            return new Reading(false, Fault.NO_MATCH, "NO FRAME",
                    Float.NaN, Float.NaN, Float.NaN, Float.NaN);
        }

        float motion = measureMotion(frame);

        float[] vector = embedder.embed(frame);
        if (vector == null) {
            return new Reading(false, Fault.NO_MATCH, "NO MATCH — POINT AT THE PART",
                    Float.NaN, Float.NaN, Float.NaN, motion);
        }
        normalise(vector);
        float similarity = dot(vector, referenceEmbedding);

        // A frame short of the threshold is rejected whatever the probes say, so skip them
        // and spare the encoder pass — that is most frames while the operator is still
        // hunting for the area.
        if (similarity < MATCH_THRESHOLD) {
            return classify(similarity, Float.NaN, Float.NaN, motion);
        }

        float closeGain = Float.NaN;
        if (referenceZoomEmbedding != null) {
            closeGain = dot(vector, referenceZoomEmbedding) - similarity;
        }

        // Only worth paying for the frame's crop when it is not already failing the cheap
        // test — the two faults are reported one at a time anyway.
        float farGain = Float.NaN;
        if (!(closeGain > SCALE_GAIN_TOLERANCE)) {
            farGain = measureFarGain(frame, similarity);
        }

        return classify(similarity, farGain, closeGain, motion);
    }

    /**
     * Re-scores the frame's centre crop against the reference. A positive result means the
     * match improves when the frame is zoomed in, i.e. the area is being shot too small.
     * Returns NaN when the crop could not be encoded, which suppresses the test rather than
     * failing the frame.
     */
    private float measureFarGain(Bitmap frame, float baseSimilarity) {
        Bitmap probe = centreCrop(frame, PROBE_CROP);
        if (probe == null) return Float.NaN;
        try {
            float[] vector = embedder.embed(probe);
            if (vector == null) return Float.NaN;
            normalise(vector);
            return dot(vector, referenceEmbedding) - baseSimilarity;
        } finally {
            if (probe != frame) probe.recycle();
        }
    }

    /** Centre crop keeping {@code fraction} of each side, or null if the source is too small. */
    private static Bitmap centreCrop(Bitmap src, float fraction) {
        if (src == null) return null;
        int w = Math.round(src.getWidth() * fraction);
        int h = Math.round(src.getHeight() * fraction);
        if (w < 8 || h < 8) return null;
        return Bitmap.createBitmap(src, (src.getWidth() - w) / 2, (src.getHeight() - h) / 2, w, h);
    }

    /** Convenience for callers with no scale probe, such as the threshold tests. */
    static Reading classify(float similarity, float motion) {
        return classify(similarity, Float.NaN, Float.NaN, motion);
    }

    /**
     * Applies the thresholds and picks the single most useful thing to tell the operator.
     *
     * <p>Distance faults outrank motion: a frame held at the wrong distance is steady-wrong,
     * and telling the operator to hold still would strand them there. NaN gains mean the
     * probe did not run and simply skip their test.
     */
    static Reading classify(float similarity, float farGain, float closeGain, float motion) {
        boolean matches = similarity >= MATCH_THRESHOLD;
        // The first frame has no predecessor to compare against, so it cannot be judged
        // moving; the hold needs several frames anyway, so nothing is let through early.
        boolean steady = Float.isNaN(motion) || motion <= MAX_MOTION;

        if (!matches) {
            if (similarity < MATCH_THRESHOLD * NO_MATCH_FRACTION) {
                return new Reading(false, Fault.NO_MATCH, "NO MATCH — POINT AT THE PART",
                        similarity, farGain, closeGain, motion);
            }
            // Close but short: the operator is looking at roughly the right thing, and
            // moving closer or squaring up is what raises the score.
            return new Reading(false, Fault.WEAK_MATCH, "CLOSE — FRAME IT LIKE THE GHOST",
                    similarity, farGain, closeGain, motion);
        }

        if (closeGain > SCALE_GAIN_TOLERANCE) {
            return new Reading(false, Fault.TOO_CLOSE, "TOO CLOSE — BACK OFF",
                    similarity, farGain, closeGain, motion);
        }
        if (farGain > SCALE_GAIN_TOLERANCE) {
            return new Reading(false, Fault.TOO_FAR, "TOO FAR — MOVE CLOSER",
                    similarity, farGain, closeGain, motion);
        }
        if (!steady) {
            return new Reading(false, Fault.MOVING, "STEADY THE PHONE",
                    similarity, farGain, closeGain, motion);
        }
        return new Reading(true, Fault.NONE, "HOLD STILL",
                similarity, farGain, closeGain, motion);
    }

    /**
     * Mean absolute luminance change against the previous frame, on a 0-255 scale.
     * Returns NaN for the first frame.
     */
    float measureMotion(Bitmap frame) {
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
