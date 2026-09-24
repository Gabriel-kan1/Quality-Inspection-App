package com.example.ml;

import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Which part of the photograph is not the chassis.
 *
 * <p>A wide shot catches the bench, the floor, a rack of stock or a tool lying beside the
 * part, and every one of those produces contours that reach the detector as candidate
 * hardware. On {@code whole screws1-3.jpg} that is a strip of cluttered floor down the left
 * edge; on {@code hole1.jpg} a shadowed region at the right that
 * {@code python/detection_ground_truth.py} already names as a spurious detection; on
 * {@code three.jpg} a dark object lying across the bottom.
 *
 * <h3>What separates chassis from background</h3>
 * Not brightness. The first attempt keyed on it and failed in the dangerous direction: a
 * specular highlight and a lighting gradient are both a long way from the plate's modal grey,
 * so it whited out 24% of {@code screws_flash.jpg} and 18% of {@code head2.jpg} — all of it
 * chassis — while missing the actual bench.
 *
 * <p>What does separate them, measured as Canny density over a margin strip against the same
 * image's centre: <b>clutter</b>. Where there is background the margin runs 2.3–6.1% edge
 * density against a centre at 0.3%; where the plate reaches the frame every margin of every
 * close-up in the asset set is at or below 0.7%. A second, independent signature covers a
 * flat dark object lying over the part ({@code three.jpg}'s bottom strip: mean 38.8 against a
 * plate at 146), which is dark rather than cluttered.
 *
 * <h3>The safety property</h3>
 * A region is only ever background if it <b>touches the frame border</b>. An interior region
 * can never be removed, so a fastener, the shadow beside one, or a dark recess in the middle
 * of the part is never mistaken for background however dark or busy it is. Verified on every
 * asset where the mask fires: of the detections it removes, none is a fastener — they are the
 * shadowed region in {@code hole1.jpg}, the bench edge in {@code hole2.jpg} and the floor
 * strips in {@code whole screws1-3.jpg}.
 *
 * <h3>Why the masked region is filled with the plate's own grey, not white</h3>
 * The operator asked for white and white is what the <em>overlay</em> shows. The pixels the
 * detector sees are filled with the median grey of everything outside the mask instead,
 * because filling with white is measurably destructive: it moves the intensity histogram, and
 * the edge stage derives Canny's thresholds from that histogram by Otsu. Scored against the
 * ground-truth counts (weighted = 2 x misses + extras over 26 photographs):
 *
 * <pre>
 *   no mask                    20/26 exact,  3 misses, 13 extras, weighted 19
 *   fill white                 17/26 exact, 33 misses,  6 extras, weighted 72
 *   fill white + cut the seam  18/26 exact, 11 misses, 18 extras, weighted 40
 *   fill the plate's grey      20/26 exact,  4 misses,  3 extras, weighted 11
 * </pre>
 *
 * White cost thirty misses — on {@code whole screws2.jpg} it took the count from 13 to 1.
 * Filling with the plate's own grey leaves the histogram where it was, raises no seam for
 * Canny to find, and cuts false positives from 13 to 3. See
 * {@code python/background_mask_study.py}.
 */
public final class ChassisMask {

    private static final String TAG = "ChassisMask";

    /** Width the mask is computed at. Cheap, and it averages the plate's texture away. */
    static final int MASK_WIDTH = 320;
    /** Box-filter window for the local clutter measure, in mask pixels. */
    static final int CLUTTER_WINDOW = 25;
    /** How many times the plate's own clutter a region must carry before it is foreign. */
    static final double CLUTTER_MULTIPLE = 4.0;
    /**
     * Absolute clutter floor. Without it a mirror-smooth plate, whose centre clutter is 0,
     * would make every faint texture anywhere "four times the plate" and mask the lot.
     */
    static final double CLUTTER_FLOOR = 0.02;
    /** Below this fraction of the plate's brightness a region is foreign however smooth. */
    static final double DARK_RATIO = 0.40;
    /** A border region smaller than this share of the frame is an edge artefact, not background. */
    static final double MIN_BACKGROUND_FRACTION = 0.01;
    /** The share of the frame, centred, assumed to be chassis and used to characterise it. */
    static final double CENTRE_FRACTION = 0.4;
    /**
     * Refuse to mask more than this. Past it the "centre is chassis" assumption has clearly
     * failed — the part is not what fills the frame — and a mask covering most of the photo
     * would be a guess with nothing behind it.
     */
    static final double MAX_BACKGROUND_FRACTION = 0.45;

    private ChassisMask() {}

    /** The mask and what it cost to find, so the caller can report and release it. */
    public static final class Result {
        /** CV_8UC1 at the source size: 255 where the pixel is not the chassis. Caller releases. */
        public final Mat mask;
        /** Share of the frame the mask covers, 0 when nothing was masked. */
        public final double fraction;
        /** Median grey of everything outside the mask — what the masked pixels are filled with. */
        public final int plateGrey;

        Result(Mat mask, double fraction, int plateGrey) {
            this.mask = mask;
            this.fraction = fraction;
            this.plateGrey = plateGrey;
        }

        public boolean isEmpty() {
            return mask == null || fraction <= 0;
        }

        public void release() {
            if (mask != null) mask.release();
        }
    }

    /**
     * Finds what is not the chassis.
     *
     * @param srcRgba working-resolution RGBA; not modified
     * @return the mask, or a result whose {@link Result#isEmpty()} is true when the whole
     *         frame reads as chassis — which is the normal case for a close-up
     */
    public static Result of(Mat srcRgba) {
        Mat gray = new Mat();
        Mat small = new Mat();
        Mat blurred = new Mat();
        Mat edges = new Mat();
        Mat edgesF = new Mat();
        Mat clutter = new Mat();
        Mat foreign = new Mat();
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        Mat kernel = null;
        Mat maskSmall = null;
        Mat full = null;
        try {
            Imgproc.cvtColor(srcRgba, gray, Imgproc.COLOR_RGBA2GRAY);
            int w0 = gray.cols(), h0 = gray.rows();
            if (w0 < 16 || h0 < 16) return new Result(null, 0, 0);
            double scale = MASK_WIDTH / (double) w0;
            Imgproc.resize(gray, small, new Size(MASK_WIDTH, Math.max(1, Math.round(h0 * scale))),
                    0, 0, Imgproc.INTER_AREA);
            int w = small.cols(), h = small.rows();

            Imgproc.GaussianBlur(small, blurred, new Size(5, 5), 0);
            Imgproc.Canny(blurred, edges, 60, 160);
            edges.convertTo(edgesF, CvType.CV_32F, 1.0 / 255.0);
            Imgproc.boxFilter(edgesF, clutter, -1, new Size(CLUTTER_WINDOW, CLUTTER_WINDOW));

            Rect centre = new Rect(
                    (int) (w * (0.5 - CENTRE_FRACTION / 2)), (int) (h * (0.5 - CENTRE_FRACTION / 2)),
                    Math.max(1, (int) (w * CENTRE_FRACTION)), Math.max(1, (int) (h * CENTRE_FRACTION)));
            double plateClutter = median(new Mat(clutter, centre), true);
            double plateGreySmall = median(new Mat(small, centre), false);
            double limit = Math.max(CLUTTER_FLOOR, CLUTTER_MULTIPLE * plateClutter);

            // Foreign = too busy to be the plate, or too dark to be it.
            Mat busy = new Mat();
            Mat dark = new Mat();
            try {
                Core.compare(clutter, new Scalar(limit), busy, Core.CMP_GT);
                Core.compare(small, new Scalar(DARK_RATIO * plateGreySmall), dark, Core.CMP_LT);
                Core.bitwise_or(busy, dark, foreign);
            } finally {
                busy.release();
                dark.release();
            }
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(15, 15));
            Imgproc.morphologyEx(foreign, foreign, Imgproc.MORPH_CLOSE, kernel);

            int n = Imgproc.connectedComponentsWithStats(foreign, labels, stats, centroids, 8, CvType.CV_32S);
            maskSmall = Mat.zeros(h, w, CvType.CV_8UC1);
            int kept = 0;
            for (int i = 1; i < n; i++) {
                int x = (int) stats.get(i, Imgproc.CC_STAT_LEFT)[0];
                int y = (int) stats.get(i, Imgproc.CC_STAT_TOP)[0];
                int bw = (int) stats.get(i, Imgproc.CC_STAT_WIDTH)[0];
                int bh = (int) stats.get(i, Imgproc.CC_STAT_HEIGHT)[0];
                double area = stats.get(i, Imgproc.CC_STAT_AREA)[0];
                if (!isBackgroundRegion(x, y, bw, bh, area, w, h)) continue;
                Mat isThis = new Mat();
                try {
                    Core.compare(labels, new Scalar(i), isThis, Core.CMP_EQ);
                    Core.bitwise_or(maskSmall, isThis, maskSmall);
                    kept++;
                } finally {
                    isThis.release();
                }
            }
            if (kept == 0) return new Result(null, 0, (int) plateGreySmall);

            double fraction = Core.countNonZero(maskSmall) / (double) (w * h);
            if (fraction > MAX_BACKGROUND_FRACTION) {
                Log.d(TAG, String.format(java.util.Locale.US,
                        "Refusing a background mask covering %.0f%% of the frame — the part is"
                                + " not what fills this photograph, so the centre assumption fails.",
                        fraction * 100));
                return new Result(null, 0, (int) plateGreySmall);
            }

            full = new Mat();
            Imgproc.resize(maskSmall, full, new Size(w0, h0), 0, 0, Imgproc.INTER_NEAREST);
            int plate = plateGreyOutside(gray, full, (int) plateGreySmall);
            Log.d(TAG, String.format(java.util.Locale.US,
                    "Background mask: %d region(s), %.1f%% of the frame, plate grey %d"
                            + " (clutter %.4f against a limit of %.3f)",
                    kept, fraction * 100, plate, plateClutter, limit));
            Mat out = full;
            full = null;
            return new Result(out, fraction, plate);
        } catch (Throwable t) {
            Log.e(TAG, "Background masking failed; treating the whole frame as chassis", t);
            return new Result(null, 0, 0);
        } finally {
            gray.release();
            small.release();
            blurred.release();
            edges.release();
            edgesF.release();
            clutter.release();
            foreign.release();
            labels.release();
            stats.release();
            centroids.release();
            if (kernel != null) kernel.release();
            if (maskSmall != null) maskSmall.release();
            if (full != null) full.release();
        }
    }

    /**
     * Whether one connected region of not-plate pixels counts as background.
     *
     * <p>Two conditions, and the first is the safety property this whole class rests on: the
     * region must <b>touch the frame border</b>. An interior region is never background, so a
     * fastener, the shadow beside one, or a dark recess in the middle of the part cannot be
     * removed however dark or busy it is. The second condition drops slivers along the frame
     * edge, which are a rounding artefact of the downscale rather than a bench.
     *
     * <p>Pure, and package-visible, so the rule can be tested without OpenCV.
     */
    static boolean isBackgroundRegion(int x, int y, int bw, int bh, double area, int w, int h) {
        boolean touchesBorder = x <= 0 || y <= 0 || (x + bw) >= w || (y + bh) >= h;
        return touchesBorder && area >= MIN_BACKGROUND_FRACTION * (double) w * h;
    }

    /**
     * Replaces the masked pixels with the plate's own median grey.
     *
     * <p>Not white: see the class comment. White moves the intensity histogram the edge stage
     * derives Canny's thresholds from, and cost thirty misses across the asset set.
     */
    public static void fillWithPlate(Mat srcRgba, Result result) {
        if (srcRgba == null || result == null || result.isEmpty()) return;
        srcRgba.setTo(new Scalar(result.plateGrey, result.plateGrey, result.plateGrey, 255),
                result.mask);
    }

    /**
     * Median of a single-channel Mat, which the caller hands over and this releases.
     *
     * <p>The input is a ROI view of a larger Mat, so it is <em>not continuous</em> and
     * {@code reshape} refuses it outright — OpenCV cannot change the row count of a matrix
     * whose rows are not adjacent in memory. Cloning first is what makes it continuous. This
     * is the one thing here the desktop mirror could not have caught, since NumPy slicing has
     * no such restriction; it surfaced on the first device run.
     */
    private static double median(Mat m, boolean asFloat) {
        Mat dense = null;
        Mat flat = null;
        Mat sorted = null;
        try {
            dense = m.isContinuous() ? m : m.clone();
            flat = dense.reshape(1, 1);
            sorted = new Mat();
            Core.sort(flat, sorted, Core.SORT_ASCENDING + Core.SORT_EVERY_ROW);
            return sorted.get(0, sorted.cols() / 2)[0];
        } finally {
            if (sorted != null) sorted.release();
            if (flat != null) flat.release();
            if (dense != null && dense != m) dense.release();
            m.release();
        }
    }

    /** Median grey of everything the mask does not cover, falling back to {@code fallback}. */
    private static int plateGreyOutside(Mat gray, Mat mask, int fallback) {
        Mat inverse = new Mat();
        Mat hist = new Mat();
        try {
            Core.bitwise_not(mask, inverse);
            java.util.List<Mat> images = new java.util.ArrayList<>();
            images.add(gray);
            Imgproc.calcHist(images, new org.opencv.core.MatOfInt(0), inverse, hist,
                    new org.opencv.core.MatOfInt(256), new org.opencv.core.MatOfFloat(0f, 256f));
            double total = 0;
            double[] counts = new double[256];
            for (int i = 0; i < 256; i++) {
                counts[i] = hist.get(i, 0)[0];
                total += counts[i];
            }
            if (total <= 0) return fallback;
            double seen = 0;
            for (int i = 0; i < 256; i++) {
                seen += counts[i];
                if (seen >= total / 2) return i;
            }
            return fallback;
        } catch (Throwable t) {
            return fallback;
        } finally {
            inverse.release();
            hist.release();
        }
    }
}
