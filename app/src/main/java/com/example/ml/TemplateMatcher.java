package com.example.ml;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Locale;

/**
 * Locates one photograph inside another and reports how the framing differs: a match score,
 * a framing ratio, and which way the camera has to move.
 *
 * <h3>Why gradients and not pixels</h3>
 *
 * Normalised cross-correlation on raw intensity does not survive this hardware. Measured
 * with exact ground truth — a known sub-region cropped from one photo of a scene, searched
 * for in a <em>different</em> photo of the same scene under different light:
 *
 * <pre>
 *   raw grayscale      0/10 recovered      mean scale error 0.275
 *   CLAHE grayscale    0/10                                 0.245
 *   Sobel magnitude   10/10                                 0.015
 *   Canny edges        3/10                                 0.185
 * </pre>
 *
 * Reproduce with {@code python python/template_matching_study.py --part A}.
 *
 * The plate is matte and specular: intensity is dominated by where the light happens to be,
 * while the gradient response is dominated by the hardware's edges, which is the content
 * that actually identifies the region. CLAHE does not rescue intensity — it amplifies the
 * same misleading shading.
 *
 * <h3>Working size</h3>
 *
 * {@link #WORK_WIDTH} 320 is the <em>smallest</em> width that recovers every case, which is the
 * reason to prefer it: search cost grows with the square of the width and wider is no more
 * accurate. Against the same ground truth: 192 px 5/10, 256 px 8/10, <b>320 px 10/10</b>,
 * 480 px 10/10, 640 px 10/10. Below 320 the downsampling takes the fastener edges with the
 * sensor noise.
 *
 * <p>An earlier run put 480 px at 8/10 and 640 px at 9/10 and this paragraph claimed 320 beat
 * both. That did not survive re-measurement against the shipped resampling; the accuracy
 * argument for 320 over wider was wrong, and the cost argument stands on its own.
 *
 * <h3>What this cannot do</h3>
 *
 * The search covers translation and uniform scale only. It has no model of rotation or
 * perspective, so it degrades as the camera angle changes rather than merely the distance.
 * Measured across real pairs shot from different angles the framing ratio is unreliable, and
 * the score does <em>not</em> separate the reference area from a different one. Worst same-area
 * against best different-area: 0.322 vs 0.386 on {@code screws.jpg}, 0.355 vs 0.507 on
 * {@code screws_flash.jpg} — negative separation both times, so no threshold divides them.
 *
 * <p>So this answers "how should the camera move from roughly here", not "is this the right
 * part". {@link AlignmentGate} keeps the encoder for the second question and uses this for
 * the first. Do not promote the score to an area test without re-running that measurement.
 *
 * <h3>Why the peak is not taken at face value</h3>
 *
 * Correlation peaks at different template sizes are not comparable. A template shrunk to 30%
 * is a hundred-pixel blob of plate texture that correlates ~0.48 with <em>something</em> in any
 * textured scene — a photograph matched against itself scores ~0.54 at that scale — while at
 * the true scale a few degrees of camera angle, which the search does not model, can drop the
 * correlation below that floor. {@code whole screws1.jpg} against {@code whole screws2.jpg},
 * the same plate from the same distance, peaked at 0.48 for scale 0.30 and 0.31 for scale
 * 1.00, and the gate told the operator they were 3.3x too far away.
 *
 * <p>{@link #SCALE_PRIOR} settles it: each scale's peak is charged {@code SCALE_PRIOR *
 * |ln scale|} before the best is chosen, so a small template's chance-level peak has to beat
 * the unchanged-framing reading by a margin that grows with how much of a change it claims,
 * while a genuine strong peak — the exact crop cases score 0.6 and up — still wins outright.
 * Measured in {@code python/framing_robustness_study.py} over the ten exact ground-truth
 * crops and every same-scene pair in the catalogue: 0.08–0.12 is a plateau that recovers one
 * more exact crop (9/10 from 8/10), turns two more gate verdicts right (117/129 from 115) and
 * breaks none; 0.15 starts to over-pull toward 1.0. Blurring the gradient maps, flooring the
 * sweep and subtracting a rotated-scene chance level were all measured too and all cost
 * verdicts. The {@link Match#score} reported is still the raw peak, which is what
 * {@code AlignmentGate.MIN_TEMPLATE_SCORE} is fitted against.
 */
public final class TemplateMatcher {

    private static final String TAG = "TemplateMatcher";

    /** Width both images are reduced to before matching. See the class notes. */
    public static final int WORK_WIDTH = 320;

    private static final double SCALE_MIN = 0.30;
    private static final double SCALE_MAX = 1.00;
    private static final double SCALE_STEP = 0.05;

    /** Smallest template edge worth correlating; below this the peak is noise. */
    private static final int MIN_TEMPLATE_EDGE = 16;

    /**
     * Penalty per unit of {@code |ln scale|} charged to a scale's peak before scales are
     * compared. Middle of the 0.08–0.12 plateau found in
     * {@code python/framing_robustness_study.py}; see the class notes for what it buys.
     */
    static final double SCALE_PRIOR = 0.10;

    /** One search result. Immutable so it can cross to the UI thread. */
    public static final class Match {
        /** Peak normalised cross-correlation, -1..1. Higher is a tighter fit. */
        public final float score;
        /**
         * The frame's field of view as a multiple of the reference's. 1.0 is the same
         * framing, below 1 means the frame covers less of the part than the reference did
         * (too close, hardware falls outside), above 1 means it covers more (too far).
         */
        public final float scale;
        /** How far the aim is off, as a fraction of the frame. Positive means pan right. */
        public final float offsetX;
        /** Positive means tilt down. */
        public final float offsetY;

        Match(float score, float scale, float offsetX, float offsetY) {
            this.score = score;
            this.scale = scale;
            this.offsetX = offsetX;
            this.offsetY = offsetY;
        }

        /** Largest single framing error, so callers can rank advice without repeating the maths. */
        public float worstOffset() {
            return Math.max(Math.abs(offsetX), Math.abs(offsetY));
        }

        @Override
        public String toString() {
            return String.format(Locale.US, "score %.3f scale %.2f offset (%+.2f,%+.2f)",
                    score, scale, offsetX, offsetY);
        }
    }

    private TemplateMatcher() {}

    /**
     * Reduces a bitmap to the gradient-magnitude image the search runs on. Returns null when
     * OpenCV is unavailable or the bitmap cannot be read; the caller must treat that as
     * "cannot judge framing" rather than as a pass.
     */
    public static Mat prepare(Bitmap source) {
        if (source == null || !HardwareFeatureExtractor.isAvailable()) return null;
        Mat rgba = new Mat();
        Mat gray = new Mat();
        Mat gx = new Mat();
        Mat gy = new Mat();
        Mat mag = new Mat();
        try {
            Utils.bitmapToMat(source, rgba);
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY);

            int w = WORK_WIDTH;
            int h = Math.max(1, Math.round(gray.rows() * (float) w / gray.cols()));
            Imgproc.resize(gray, gray, new Size(w, h), 0, 0, Imgproc.INTER_AREA);
            Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

            Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0, 3);
            Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1, 3);
            Core.magnitude(gx, gy, mag);

            Mat out = new Mat();
            Core.normalize(mag, out, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
            return out;
        } catch (Throwable t) {
            Log.e(TAG, "prepare() failed", t);
            return null;
        } finally {
            rgba.release();
            gray.release();
            gx.release();
            gy.release();
            mag.release();
        }
    }

    /**
     * Finds where {@code frame} sits relative to {@code reference}.
     *
     * <p>Searched both ways round because either image may be the larger view: the frame
     * inside the reference when the operator is too close, the reference inside the frame
     * when they are too far. The direction that correlates better wins, which is also what
     * decides the sign of the framing ratio.
     *
     * @return the match, or null when either image is missing or too small to search.
     */
    public static Match match(Mat reference, Mat frame) {
        if (reference == null || frame == null || reference.empty() || frame.empty()) return null;

        double[] closer = searchIn(reference, frame);   // frame is part of the reference
        double[] wider = searchIn(frame, reference);    // reference is part of the frame
        if (closer == null && wider == null) return null;

        // Compared on the ranked score, so the prior decides between the two directions the
        // same way it decides between scales within one; the Match reports the raw peak.
        boolean useCloser = wider == null || (closer != null && closer[0] >= wider[0]);
        double[] best = useCloser ? closer : wider;

        if (useCloser) {
            // Frame occupies best[1] of the reference, centred at (best[2], best[3]) in
            // reference coordinates. Aiming at the reference's middle means moving toward it.
            return new Match((float) best[4], (float) best[1],
                    (float) (0.5 - best[2]), (float) (0.5 - best[3]));
        }
        // The reference occupies best[1] of the frame, so the frame covers 1/best[1] of it,
        // and the reference sits at (best[2], best[3]) in the operator's view — move toward it.
        float scale = best[1] > 0 ? (float) (1.0 / best[1]) : Float.NaN;
        return new Match((float) best[4], scale,
                (float) (best[2] - 0.5), (float) (best[3] - 0.5));
    }

    /**
     * A scale's peak with the prior charged: what scales are ranked by. Pure, so the rule
     * that decides between "unchanged" and "a third of the size" can be tested on its own.
     */
    static double ranked(double peak, double scale) {
        return peak - SCALE_PRIOR * Math.abs(Math.log(scale));
    }

    /**
     * Slides {@code templ} over {@code scene} at every scale in the sweep and keeps the scale
     * with the best {@link #ranked} score.
     *
     * @return {ranked score, scale, centreX, centreY, raw peak} with the centre as a fraction
     *         of the scene, or null when no scale produced a usable template.
     */
    private static double[] searchIn(Mat scene, Mat templ) {
        double[] best = null;
        double aspect = templ.rows() / (double) templ.cols();

        for (double s = SCALE_MIN; s <= SCALE_MAX + 1e-9; s += SCALE_STEP) {
            int tw = (int) Math.round(scene.cols() * s);
            int th = (int) Math.round(tw * aspect);
            if (tw < MIN_TEMPLATE_EDGE || th < MIN_TEMPLATE_EDGE
                    || tw > scene.cols() || th > scene.rows()) {
                continue;
            }

            Mat resized = new Mat();
            Mat result = new Mat();
            try {
                Imgproc.resize(templ, resized, new Size(tw, th), 0, 0, Imgproc.INTER_AREA);
                Imgproc.matchTemplate(scene, resized, result, Imgproc.TM_CCOEFF_NORMED);
                Core.MinMaxLocResult mm = Core.minMaxLoc(result);
                double score = ranked(mm.maxVal, s);
                if (best == null || score > best[0]) {
                    best = new double[]{
                            score,
                            s,
                            (mm.maxLoc.x + tw / 2.0) / scene.cols(),
                            (mm.maxLoc.y + th / 2.0) / scene.rows(),
                            mm.maxVal};
                }
            } catch (Throwable t) {
                Log.w(TAG, "matchTemplate failed at scale " + s, t);
            } finally {
                resized.release();
                result.release();
            }
        }
        return best;
    }
}
