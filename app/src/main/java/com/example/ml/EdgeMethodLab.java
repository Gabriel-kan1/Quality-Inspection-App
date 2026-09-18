package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Runs one image through several edge/feature operators and reports what each one produced.
 *
 * <p>This exists because the detector's edge stage was chosen once, in Python, against a
 * handful of photographs, and every tuning argument since has been conducted on assertion
 * rather than evidence. The pipeline that ships is a specific stack — glare correction, CLAHE,
 * a bilateral filter, two blurs, Otsu-scaled Canny, then dilate and close — and no screen in
 * the app has ever shown what the simpler alternatives would have done with the same photo.
 *
 * <p>Two rules keep the comparison honest:
 *
 * <ul>
 *   <li><b>One ruler.</b> Every method sees the same image at
 *       {@link HardwareFeatureExtractor#WORKING_MAX_DIM}, because the kernel sizes and area
 *       floors downstream are denominated in pixels at that resolution. Comparing methods
 *       across two source resolutions would measure the resize, not the operator.
 *   <li><b>One scorer.</b> Every edge map is pushed through the detector's own
 *       {@link HardwareFeatureExtractor#findFeatureBoxes} geometry filter. Ranking by raw
 *       contour count would crown whichever method produced the most noise.
 * </ul>
 *
 * <p>The shipped pipeline is included as one of the entries and is invoked through
 * {@link HardwareFeatureExtractor#computeEdges} rather than copied, so this screen cannot
 * drift away from the code it is supposed to be evidence about. Five tuned variants of it
 * follow — see {@code python/hybrid_v2_study.py}, which is where they were found and where
 * the numbers quoted in their descriptions come from.
 *
 * <p><b>No method is declared the winner.</b> The counts are reported and left to be read:
 * more candidate boxes is not better when the extra ones are seams and shadows, and the only
 * way to know which they are is to look at the maps next to each other.
 *
 * <h3>Choosing one</h3>
 * The operator can pick any catalogued method to be the edge stage that inspections run
 * ({@link #select}); the choice persists across launches and defaults to the shipped stack.
 * {@link HardwareFeatureExtractor#computeEdges(Context, Mat, Boolean)} consults it for every
 * image, reference setup and inspection alike, so the two sides of a count comparison always
 * come through the same operator. A method that is not the shipped stack hands the pipeline
 * one map for all three of its roles — contour finding, the diagnostic overlay and marker
 * measurement — because that is what the lab scored and what the operator looked at; a
 * variant that does its own dilate/close therefore measures a little large in millimetres,
 * exactly as the shipped closed map would.
 *
 * <p>A reference layout records the method it was labelled with. Changing the method
 * afterwards is a detector change — the same fastener can come out as one box or two — and
 * the results screen says so rather than comparing the counts as if nothing had moved.
 */
public final class EdgeMethodLab {

    private static final String TAG = "EdgeMethodLab";

    /** Where the chosen method lives. One preference file, one key, one id. */
    static final String PREFS = "edge_method";
    static final String KEY_SELECTED = "selected";
    /** Catalogue id of the stack the detector was tuned with; what runs until someone chooses. */
    public static final String SHIPPED_ID = "shipped-hybrid";

    /**
     * Longest edge of the edge-map thumbnails handed back for display.
     *
     * <p>A dozen maps at the 1280px working resolution would be about 60MB of bitmap in
     * native memory at once, which on Android 8+ does not raise OutOfMemoryError — it grows
     * the process until the kernel kills it. Metrics are still measured at full working
     * resolution; only the picture is shrunk.
     */
    static final int DISPLAY_MAX_DIM = 640;

    /** Gaussian window shared by the operators that need smoothing before differentiating. */
    private static final Size BLUR_5 = new Size(5, 5);

    // Aliased from the detector rather than copied, so a hybrid variant claiming to differ
    // from production in four values cannot quietly differ in a fifth.
    private static final double GLARE_RATIO_THRESHOLD =
            HardwareFeatureExtractor.GLARE_RATIO_THRESHOLD;
    private static final int GLARE_KERNEL_SIZE = HardwareFeatureExtractor.GLARE_KERNEL_SIZE;
    private static final double CLAHE_CLIP_LIMIT = HardwareFeatureExtractor.CLAHE_CLIP_LIMIT;
    private static final double BILATERAL_SIGMA_COLOR =
            HardwareFeatureExtractor.BILATERAL_SIGMA_COLOR;
    private static final double BILATERAL_SIGMA_SPACE =
            HardwareFeatureExtractor.BILATERAL_SIGMA_SPACE;
    private static final double CANNY_LOW_RATIO = HardwareFeatureExtractor.CANNY_LOW_RATIO;
    private static final int DILATE_KERNEL_SIZE = HardwareFeatureExtractor.DILATE_KERNEL_SIZE;
    private static final int CLOSE_KERNEL_SIZE = HardwareFeatureExtractor.CLOSE_KERNEL_SIZE;

    /**
     * Canny's Sobel aperture. Not aliased because production never names it: it calls the
     * four-argument overload and takes the default, which is 3. Stated explicitly here only
     * because the six-argument overload needed to reach {@code L2gradient} demands it.
     */
    private static final int CANNY_APERTURE = 3;

    private EdgeMethodLab() {}

    /** One operator's output and what it cost. */
    public static final class MethodResult {
        /** Stable catalogue id, the value {@link #select} takes. */
        public final String id;
        public final String name;
        /** What the operator does, and what to look for in its map. */
        public final String summary;
        /** Binary edge map, downscaled for display; null when the method failed. */
        public final Bitmap edgeMap;
        /** Percentage of pixels the operator marked as edge, at working resolution. */
        public final double edgeDensityPercent;
        /** External contours found in the map, before any geometry filtering. */
        public final int contourCount;
        /** Contours surviving the detector's own aspect/area/nesting filter. */
        public final int candidateBoxes;
        /** Wall time for the operator alone, excluding scoring and display conversion. */
        public final long elapsedMs;
        /** True for the stack the app actually ships, so the UI can mark it. */
        public final boolean isShippedPipeline;
        /** True for the method inspections currently run — see {@link #selectedId}. */
        public final boolean inUse;
        /** Non-null when the operator threw; the other numbers are meaningless then. */
        public final String failure;

        MethodResult(String id, String name, String summary, Bitmap edgeMap,
                     double edgeDensityPercent, int contourCount, int candidateBoxes,
                     long elapsedMs, boolean isShippedPipeline, boolean inUse, String failure) {
            this.id = id;
            this.inUse = inUse;
            this.name = name;
            this.summary = summary;
            this.edgeMap = edgeMap;
            this.edgeDensityPercent = edgeDensityPercent;
            this.contourCount = contourCount;
            this.candidateBoxes = candidateBoxes;
            this.elapsedMs = elapsedMs;
            this.isShippedPipeline = isShippedPipeline;
            this.failure = failure;
        }

        public boolean failed() {
            return failure != null;
        }

        /** One line of numbers for the card, in the order they are worth reading. */
        public String metricsLine() {
            if (failed()) return "Did not run: " + failure;
            return String.format(Locale.US,
                    "%d candidates  ·  %d contours  ·  %.2f%% edge  ·  %d ms",
                    candidateBoxes, contourCount, edgeDensityPercent, elapsedMs);
        }
    }

    /** An operator, named and described, that turns an image into a binary edge map. */
    interface Operator {
        /**
         * @param gray       working-resolution grayscale, owned by the caller — do not release
         * @param srcRgba    the same image in RGBA, for stages that start from colour
         * @param forceGlare pins the black-hat glare branch on or off for methods that have
         *                   one; null lets the method decide per image. Methods without a
         *                   glare branch ignore it. See {@code extractFeatures} for why two
         *                   images being compared must agree on it.
         * @param glareOut   one-element array; a method with a glare branch writes whether it
         *                   ran, so the caller can pin the next image to the same decision
         * @return a fresh CV_8UC1 map of 0/255, which the caller releases
         */
        Mat apply(Mat gray, Mat srcRgba, Boolean forceGlare, boolean[] glareOut);
    }

    /** Catalogue entry: what the method is called, what it does, and how to run it. */
    static final class Method {
        /** Stable id, persisted by {@link #select}; never renamed once shipped. */
        final String id;
        final String name;
        final String summary;
        final boolean shipped;
        final Operator operator;

        Method(String id, String name, String summary, boolean shipped, Operator operator) {
            this.id = id;
            this.name = name;
            this.summary = summary;
            this.shipped = shipped;
            this.operator = operator;
        }
    }

    /**
     * The methods compared, in the order they are shown.
     *
     * <p>Ordered as an argument rather than alphabetically: the bare operators first, so the
     * later entries read as answers to what the earlier ones got wrong; then the shipped
     * stack, where its extra cost can be weighed against what the cheap methods already
     * managed; then the tuned variants proposed to replace it.
     */
    static List<Method> catalogue() {
        List<Method> methods = new ArrayList<>();

        methods.add(new Method("canny-fixed", "Canny (fixed thresholds)",
                "Textbook Canny at 50/150 on the raw grayscale, with no preprocessing at all. "
                        + "The baseline every other entry has to beat. Fixed thresholds are "
                        + "what makes it fragile: the pair that finds a bolt head in even light "
                        + "goes blank on an underexposed photo and solid on a bright one.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> {
                    Mat out = new Mat();
                    Imgproc.Canny(gray, out, 50, 150);
                    return out;
                }));

        methods.add(new Method("canny-otsu", "Canny (Otsu-adaptive)",
                "The same operator with its two thresholds derived per image from Otsu's "
                        + "method instead of hard-coded, after a 5x5 Gaussian. This is the "
                        + "single change that lets one setting survive a change of lighting, "
                        + "and the difference against the row above is what it is worth.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> {
                    Mat blurred = new Mat();
                    Mat otsuMask = new Mat();
                    Mat out = new Mat();
                    try {
                        Imgproc.GaussianBlur(gray, blurred, BLUR_5, 0);
                        double high = Imgproc.threshold(blurred, otsuMask, 0, 255,
                                Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
                        Imgproc.Canny(blurred, out, high * 0.5, high);
                        return out;
                    } finally {
                        blurred.release();
                        otsuMask.release();
                    }
                }));

        methods.add(new Method("sobel", "Sobel gradient magnitude",
                "First derivatives in x and y from a 3x3 Sobel kernel, combined into a "
                        + "gradient magnitude and cut at Otsu. No non-maximum suppression and "
                        + "no hysteresis, so edges come out several pixels thick and every weak "
                        + "gradient survives — expect fat outlines and a high edge density.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> gradientMagnitude(gray, false)));

        methods.add(new Method("scharr", "Scharr gradient magnitude",
                "Sobel's 3x3 kernel is a coarse approximation of a derivative and is "
                        + "noticeably wrong on diagonals; Scharr is the 3x3 kernel with the "
                        + "best rotational symmetry. Worth reading directly against the Sobel "
                        + "row, because fastener heads are round and their gradients point "
                        + "every way at once.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> gradientMagnitude(gray, true)));

        methods.add(new Method("laplacian-of-gaussian", "Laplacian of Gaussian",
                "Second derivative after a Gaussian, rectified and cut at Otsu. Responds to "
                        + "intensity ridges rather than to steps, which picks up the machined "
                        + "crown of a bolt head that a first-derivative operator can miss — and "
                        + "amplifies sensor noise for exactly the same reason.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> {
                    Mat blurred = new Mat();
                    Mat lap = new Mat();
                    Mat absLap = new Mat();
                    Mat out = new Mat();
                    try {
                        Imgproc.GaussianBlur(gray, blurred, BLUR_5, 0);
                        Imgproc.Laplacian(blurred, lap, CvType.CV_16S, 3);
                        Core.convertScaleAbs(lap, absLap);
                        Imgproc.threshold(absLap, out, 0, 255,
                                Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
                        return out;
                    } finally {
                        blurred.release();
                        lap.release();
                        absLap.release();
                    }
                }));

        methods.add(new Method("morphological-gradient", "Morphological gradient",
                "Dilation minus erosion over a 3x3 ellipse — the local intensity range, which "
                        + "is large exactly where a boundary sits. Costs no convolution and no "
                        + "threshold tuning, and it is the cheapest thing on this list, so it "
                        + "is here to answer whether the expensive stack is buying anything.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> {
                    Mat kernel = Imgproc.getStructuringElement(
                            Imgproc.MORPH_ELLIPSE, new Size(3, 3));
                    Mat grad = new Mat();
                    Mat out = new Mat();
                    try {
                        Imgproc.morphologyEx(gray, grad, Imgproc.MORPH_GRADIENT, kernel);
                        Imgproc.threshold(grad, out, 0, 255,
                                Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
                        return out;
                    } finally {
                        kernel.release();
                        grad.release();
                    }
                }));

        methods.add(new Method("adaptive-threshold", "Adaptive threshold (Gaussian)",
                "Not a gradient operator: it segments each pixel against a local mean, so what "
                        + "comes out is filled dark regions rather than outlines. Included "
                        + "because uneven workshop lighting is the failure this app keeps "
                        + "meeting and this is the standard answer to it — read the contour "
                        + "count here as a region count, not as an edge count.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> {
                    Mat blurred = new Mat();
                    Mat out = new Mat();
                    try {
                        Imgproc.GaussianBlur(gray, blurred, BLUR_5, 0);
                        Imgproc.adaptiveThreshold(blurred, out, 255,
                                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                                Imgproc.THRESH_BINARY_INV, 25, 5);
                        return out;
                    } finally {
                        blurred.release();
                    }
                }));

        methods.add(new Method("clahe-bilateral-canny", "CLAHE + bilateral + Canny",
                "The shipped stack with the glare branch, the two blurs and the morphology all "
                        + "removed: local contrast equalisation, an edge-preserving denoise, "
                        + "then Otsu-scaled Canny. Isolates what the contrast work contributes, "
                        + "so the row below can be read as the cost of everything else.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> {
                    Mat equalised = new Mat();
                    Mat filtered = new Mat();
                    Mat otsuMask = new Mat();
                    Mat out = new Mat();
                    try {
                        CLAHE clahe = Imgproc.createCLAHE(1.0, new Size(16, 16));
                        clahe.apply(gray, equalised);
                        Imgproc.bilateralFilter(equalised, filtered, 5, 60, 20);
                        double high = Imgproc.threshold(filtered, otsuMask, 0, 255,
                                Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
                        Imgproc.Canny(filtered, out, high * 0.5, high);
                        return out;
                    } finally {
                        equalised.release();
                        filtered.release();
                        otsuMask.release();
                    }
                }));

        methods.add(new Method(SHIPPED_ID, "Shipped hybrid pipeline",
                "What the detector actually runs: black-hat glare correction when the frame is "
                        + "blown out, CLAHE, a noise-adaptive bilateral filter, median and "
                        + "Gaussian blurs, Otsu-scaled Canny, then dilate and close to bridge a "
                        + "fragmented outline into one contour. Called directly rather than "
                        + "reproduced, so this row cannot drift away from the shipped code.",
                true,
                (gray, srcRgba, forceGlare, glareOut) -> {
                    HardwareFeatureExtractor.Edges edges =
                            HardwareFeatureExtractor.computeEdges(srcRgba, forceGlare);
                    try {
                        glareOut[0] = edges.glareCorrectionApplied;
                        // The closed map is the one localisation consumes, so it is the one
                        // worth showing. edgesRaw would flatter the method by hiding the
                        // dilation that measurement has to compensate for elsewhere.
                        return edges.edgesClosed.clone();
                    } finally {
                        edges.release();
                    }
                }));

        // The tuned replacements, after the stack they are proposed against. Each hits the
        // six calibration counts exactly; they differ in how well they hold those counts when
        // the photograph or the parameters move. All three drop Canny's low threshold to 0.4x
        // the high one — every configuration that survived the wider search does, which makes
        // production's 0.5 the single clearest thing the study found wrong.
        // See python/hybrid_v2_study.py for the search and every number quoted below.
        methods.add(new Method("hybrid-v2", "Hybrid v2 (denoise + low-ratio 0.4)",
                "Pins the bilateral diameter at 9 and drops Canny's low threshold to 0.4x "
                        + "its high one instead of 0.5x, with the high one at 1.1x Otsu. The "
                        + "reliable one: one of only two configurations found that hold all "
                        + "six calibration counts under every photographic variation tested — "
                        + "dimming, brightening, gamma either way, two noise levels and JPEG "
                        + "recompression — and it barely moves when its own parameters are "
                        + "nudged. Group stability 0.26 against the shipped stack's 0.64, and "
                        + "cheaper than it. Against the full ground truth it is exact on 12 "
                        + "of 26 photographs, with 5 misses — the most of any variant.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> hybridVariant(srcRgba, forceGlare, glareOut, 9, 5, 5, 1.1, 0.4, false, 5, 7)));

        methods.add(new Method("hybrid-v3", "Hybrid v3 (wide Gaussian, best all-round)",
                "Bilateral 5, a wider Gaussian at 7, Canny at 1.2x Otsu with the same 0.4 "
                        + "low ratio. Best all-rounder in the search — it took the top combined "
                        + "rank across all three robustness tests without winning any of them "
                        + "outright: near-perfect under photographic variation, on a broad "
                        + "parameter plateau, group stability 0.18, and the cheapest. Against "
                        + "the full ground truth: 12 of 26 exact, 3 misses, 33 extras.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> hybridVariant(srcRgba, forceGlare, glareOut, 5, 5, 7, 1.2, 0.4, false, 5, 7)));

        methods.add(new Method("hybrid-v4", "Hybrid v4 (L2 gradient, best stability)",
                "Bilateral 7, the 0.4 low ratio, a true Euclidean gradient magnitude rather "
                        + "than the |Gx|+|Gy| approximation, and lighter morphology — a 3px "
                        + "dilation closed with a 9px ellipse. Best group stability of any "
                        + "configuration tested (0.13), which is the number that predicts false "
                        + "FAILs. Against the full ground truth it was far ahead of v2, v3 and "
                        + "v5 all along: 16 of 26 exact, 3 misses, 15 extras.",
                false,
                (gray, srcRgba, forceGlare, glareOut) -> hybridVariant(srcRgba, forceGlare, glareOut, 7, 5, 5, 1.0, 0.4, true, 3, 9)));

        methods.add(new Method("hybrid-v5", "Hybrid v5 (adaptive-threshold mask)",
                "v3's pipeline with an adaptive-threshold mask over the Canny map: only edges "
                        + "lying on or beside a locally dark region are kept, which is what a "
                        + "fastener is against a machined plate. The answer to whether adaptive "
                        + "thresholding helps the hybrids on the proxy tests — it does, as a "
                        + "mask and not as a detector. Against the full ground truth it does "
                        + "not: 12 of 26 exact, no better than v3, at twice the cost. Kept so "
                        + "the negative result can be seen rather than taken on trust.",
                false,
                (gray, srcRgba, forceGlare, glareOut) ->
                        hybridVariant(srcRgba, forceGlare, glareOut, 5, 5, 7, 1.2, 0.4, false, 5, 7, 61, 2, 5)));

        methods.add(new Method("hybrid-v6", "Hybrid v6 (best against ground truth)",
                "Bilateral 5, Canny at 1.1x Otsu with the L2 gradient, dilate 5 and close 9. "
                        + "Found by searching 4,032 configurations against a hand-made count "
                        + "for every bundled photograph rather than six of them, ranked on "
                        + "misses twice over extras and forbidden from emptying any image. "
                        + "Exact on 20 of 26 reference-distance photographs in the study and "
                        + "18 of 26 on this device's OpenCV, and on every one of the fourteen "
                        + "stud and screw shots in the study. Pays for it with a narrower "
                        + "parameter plateau and worse group stability than v4 (0.18 against "
                        + "0.13) — v4 remains the pick when agreement between interchangeable "
                        + "photographs matters more than absolute accuracy.",
                false,
                (gray, srcRgba, forceGlare, glareOut) ->
                        outlineEdges(srcRgba, forceGlare, glareOut)));

        return Collections.unmodifiableList(methods);
    }

    /**
     * The edge map the inspection screen draws as the reference ghost's outline.
     *
     * <p>This is Hybrid v6 — the configuration that was exact on every stud and screw
     * photograph in the ground-truth set and the most accurate overall — and it is the one
     * place v6's parameters are written down, so the Edge Lab card and the ghost outline
     * cannot drift apart. It is used here as a picture, not a verdict: the detector still
     * runs the shipped stack, and a cleaner map is simply a cleaner guide to frame against.
     *
     * @param srcRgba the reference photograph at working resolution, RGBA
     * @return a fresh CV_8UC1 map of 0/255, which the caller releases
     */
    public static Mat outlineEdges(Mat srcRgba) {
        // The ghost is one image with nothing to agree with, so the glare branch decides for
        // itself and nobody needs to know what it decided.
        return outlineEdges(srcRgba, null, new boolean[1]);
    }

    /** As above, with the glare branch pinned or reported as the catalogue's operators are. */
    static Mat outlineEdges(Mat srcRgba, Boolean forceGlare, boolean[] glareOut) {
        return hybridVariant(srcRgba, forceGlare, glareOut, 5, 5, 5, 1.1, 0.4, true, 5, 9);
    }

    /**
     * The shipped stack with the four knobs the tuning study actually moved.
     *
     * <p>A parameterised copy rather than a call into
     * {@link HardwareFeatureExtractor#computeEdges}: these are candidate <em>replacements</em>
     * for that method, so they cannot be expressed as calls to it. The shipped row above still
     * invokes the real thing, which is what keeps the comparison honest.
     *
     * <p>Everything not named here is deliberately identical to production — the same glare
     * branch and threshold, the same CLAHE clip and tile grid, the same bilateral sigmas, the
     * same dilate and close kernels — so a difference in the output is a difference in these
     * four values and nothing else.
     *
     * @param bilateralD    bilateral diameter, pinned instead of bucketed from Laplacian
     *                      variance. The bucket was the defect: it chose d=1 for the two
     *                      photographs that over-detected.
     * @param medianK       median blur aperture (odd)
     * @param gaussK        Gaussian blur aperture (odd)
     * @param otsuHighScale multiplier on the Otsu threshold used as Canny's high bound
     * @param lowRatio      Canny's low bound as a fraction of the high one. Production uses
     *                      0.5; every configuration that survived the wider search uses 0.4,
     *                      which is the single strongest signal the study produced.
     * @param l2Gradient    true for a Euclidean gradient magnitude rather than |Gx|+|Gy|
     * @param dilateK       rectangular dilation kernel that bridges a fragmented outline
     * @param closeK        elliptical closing kernel that seals it into one contour
     */
    private static Mat hybridVariant(Mat srcRgba, Boolean forceGlare, boolean[] glareOut,
                                     int bilateralD, int medianK, int gaussK,
                                     double otsuHighScale, double lowRatio, boolean l2Gradient,
                                     int dilateK, int closeK) {
        return hybridVariant(srcRgba, forceGlare, glareOut, bilateralD, medianK, gaussK,
                otsuHighScale, lowRatio, l2Gradient, dilateK, closeK, 0, 0, 0);
    }

    /**
     * As above, plus an optional {@code cv2.adaptiveThreshold} mask applied to the Canny map.
     *
     * <p>Adaptive thresholding is a poor edge detector on its own — it is in the catalogue
     * above and finds 199 candidates on the photograph the shipped stack finds 42 in. As a
     * <em>mask</em> it is a different proposition: a fastener is a locally dark region on a
     * machined plate, so keeping only the Canny edges that fall on or beside one throws away
     * the texture speckle without touching the outlines.
     *
     * <p>The mask is dilated before it is used. An edge sits on the boundary of the dark
     * region rather than inside it, so an undilated mask deletes exactly the edges it is
     * supposed to keep.
     *
     * @param maskBlock adaptiveThreshold neighbourhood, odd; 0 or less disables the mask
     * @param maskC     constant subtracted from the local Gaussian mean
     * @param maskGrow  ellipse used to dilate the mask so boundary edges survive it
     */
    private static Mat hybridVariant(Mat srcRgba, Boolean forceGlare, boolean[] glareOut,
                                     int bilateralD, int medianK, int gaussK,
                                     double otsuHighScale, double lowRatio, boolean l2Gradient,
                                     int dilateK, int closeK,
                                     int maskBlock, int maskC, int maskGrow) {
        Mat gray = new Mat();
        Mat preprocessed = new Mat();
        Mat equalised = new Mat();
        Mat filtered = new Mat();
        Mat smoothed = new Mat();
        Mat otsuMask = new Mat();
        Mat edges = new Mat();
        Mat dilateKernel = null;
        Mat closeKernel = null;
        Mat dilated = new Mat();
        Mat closed = new Mat();
        boolean ok = false;
        try {
            Imgproc.cvtColor(srcRgba, gray, Imgproc.COLOR_RGBA2GRAY);

            // Same decision, same pinning rule as the shipped stack, so a reference and an
            // inspection compared by count cannot disagree on lighting alone.
            boolean applyGlare = HardwareFeatureExtractor.decideGlareCorrection(gray, forceGlare);
            if (glareOut != null && glareOut.length > 0) glareOut[0] = applyGlare;

            if (applyGlare) {
                Mat lightingKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                        new Size(GLARE_KERNEL_SIZE, GLARE_KERNEL_SIZE));
                Imgproc.morphologyEx(gray, preprocessed, Imgproc.MORPH_BLACKHAT, lightingKernel);
                lightingKernel.release();
            } else {
                gray.copyTo(preprocessed);
            }

            CLAHE clahe = Imgproc.createCLAHE(CLAHE_CLIP_LIMIT, new Size(16, 16));
            clahe.apply(preprocessed, equalised);

            Imgproc.bilateralFilter(equalised, filtered, bilateralD,
                    BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE);

            Imgproc.medianBlur(filtered, smoothed, medianK);
            Imgproc.GaussianBlur(smoothed, smoothed, new Size(gaussK, gaussK), 0);

            double otsuHigh = Imgproc.threshold(smoothed, otsuMask, 0, 255,
                    Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU) * otsuHighScale;
            Imgproc.Canny(smoothed, edges, otsuHigh * lowRatio, otsuHigh,
                    CANNY_APERTURE, l2Gradient);

            if (maskBlock > 0) {
                Mat mask = new Mat();
                Mat growKernel = null;
                try {
                    Imgproc.adaptiveThreshold(smoothed, mask, 255,
                            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV,
                            maskBlock, maskC);
                    if (maskGrow > 0) {
                        growKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                                new Size(maskGrow, maskGrow));
                        Imgproc.dilate(mask, mask, growKernel);
                    }
                    Core.bitwise_and(edges, mask, edges);
                } finally {
                    mask.release();
                    if (growKernel != null) growKernel.release();
                }
            }

            dilateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                    new Size(dilateK, dilateK));
            Imgproc.dilate(edges, dilated, dilateKernel);

            closeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                    new Size(closeK, closeK));
            Imgproc.morphologyEx(dilated, closed, Imgproc.MORPH_CLOSE, closeKernel);

            ok = true;
            return closed;
        } finally {
            gray.release();
            preprocessed.release();
            equalised.release();
            filtered.release();
            smoothed.release();
            otsuMask.release();
            edges.release();
            dilated.release();
            if (dilateKernel != null) dilateKernel.release();
            if (closeKernel != null) closeKernel.release();
            // Only freed when unwinding on an exception; otherwise the caller owns it.
            if (!ok) closed.release();
        }
    }

    // ── The method inspections run ─────────────────────────────────────────

    /** Id of the method inspections currently run; {@link #SHIPPED_ID} until one is chosen. */
    public static String selectedId(Context context) {
        String id = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString(KEY_SELECTED, SHIPPED_ID);
        // An id from a build that had a method this one does not is not an error worth a
        // crash, and silently running an arbitrary method would be worse: fall back to the
        // stack the detector was tuned with.
        return byId(id) == null ? SHIPPED_ID : id;
    }

    /** Display name of the method inspections currently run. */
    public static String selectedName(Context context) {
        Method m = byId(selectedId(context));
        return m == null ? "Shipped hybrid pipeline" : m.name;
    }

    /** Display name for a catalogue id, or the id itself when it is not catalogued. */
    public static String nameOf(String id) {
        Method m = byId(id);
        return m == null ? String.valueOf(id) : m.name;
    }

    /**
     * Makes {@code id} the edge stage every subsequent inspection and reference setup runs.
     * Unknown ids are ignored — there is nothing sensible to do with one.
     */
    public static void select(Context context, String id) {
        if (byId(id) == null) {
            Log.w(TAG, "Ignoring unknown edge method id " + id);
            return;
        }
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putString(KEY_SELECTED, id).apply();
        Log.i(TAG, "Inspections now use edge method " + id);
    }

    /** True when the chosen method is the shipped stack, i.e. nothing has been overridden. */
    public static boolean shippedSelected(Context context) {
        return SHIPPED_ID.equals(selectedId(context));
    }

    static Method byId(String id) {
        if (id == null) return null;
        for (Method m : catalogue()) if (m.id.equals(id)) return m;
        return null;
    }

    /** The catalogue entry inspections run; never null. */
    static Method selected(Context context) {
        Method m = byId(selectedId(context));
        return m != null ? m : byId(SHIPPED_ID);
    }

    /**
     * Runs a catalogued method as the pipeline's edge stage.
     *
     * <p>The pipeline wants three maps — raw edges for marker measurement, dilated edges for
     * the diagnostic overlay, closed edges for contour finding — and the shipped stack
     * produces exactly those. Every other method produces one map, which is the map the lab
     * scored and the operator chose, so that one map is handed over for all three roles
     * rather than being pushed through a morphology the operator never saw.
     *
     * @return the maps, owned by the caller, or null when the method threw
     */
    static HardwareFeatureExtractor.Edges computeWith(Method method, Mat src,
                                                      Boolean forceGlare) {
        if (method.shipped) return HardwareFeatureExtractor.computeEdges(src, forceGlare);
        Mat gray = new Mat();
        Mat map = null;
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            boolean[] glare = new boolean[1];
            map = method.operator.apply(gray, src, forceGlare, glare);
            if (map == null) throw new IllegalStateException(method.name + " produced no map");
            HardwareFeatureExtractor.Edges edges = new HardwareFeatureExtractor.Edges(
                    gray, map, map.clone(), map.clone(), glare[0]);
            gray = null;
            map = null;
            return edges;
        } catch (Throwable t) {
            Log.e(TAG, "Edge method " + method.id + " failed; nothing detected", t);
            return null;
        } finally {
            if (gray != null) gray.release();
            if (map != null) map.release();
        }
    }

    /**
     * How many methods a run will produce, for a caller that wants to say so before starting.
     *
     * <p>Public where {@link #catalogue()} is not: the operators themselves are an
     * implementation detail of this class, but a screen announcing the wait is entitled to
     * know how long the list is without hard-coding a number that would silently go stale.
     */
    public static int catalogueSize() {
        return catalogue().size();
    }

    /**
     * Sobel and Scharr differ only in the kernel, so they share a body — otherwise the two
     * rows would be comparing an implementation difference as well as a kernel difference.
     */
    private static Mat gradientMagnitude(Mat gray, boolean scharr) {
        Mat blurred = new Mat();
        Mat gx = new Mat();
        Mat gy = new Mat();
        Mat absGx = new Mat();
        Mat absGy = new Mat();
        Mat magnitude = new Mat();
        Mat out = new Mat();
        try {
            Imgproc.GaussianBlur(gray, blurred, BLUR_5, 0);
            if (scharr) {
                Imgproc.Scharr(blurred, gx, CvType.CV_16S, 1, 0);
                Imgproc.Scharr(blurred, gy, CvType.CV_16S, 0, 1);
            } else {
                Imgproc.Sobel(blurred, gx, CvType.CV_16S, 1, 0, 3);
                Imgproc.Sobel(blurred, gy, CvType.CV_16S, 0, 1, 3);
            }
            Core.convertScaleAbs(gx, absGx);
            Core.convertScaleAbs(gy, absGy);
            // The L1 approximation OpenCV's own tutorials use. sqrt(gx^2 + gy^2) is the true
            // magnitude, but it saturates an 8-bit destination on strong edges, and a
            // saturated gradient is one Otsu cannot separate from its neighbours.
            Core.addWeighted(absGx, 0.5, absGy, 0.5, 0, magnitude);
            Imgproc.threshold(magnitude, out, 0, 255,
                    Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
            return out;
        } finally {
            blurred.release();
            gx.release();
            gy.release();
            absGx.release();
            absGy.release();
            magnitude.release();
        }
    }

    /**
     * Runs every catalogued method over one image.
     *
     * <p>Returns an empty list when OpenCV is unavailable or the image cannot be read. A
     * method that throws comes back as a {@link MethodResult} carrying its failure rather than
     * being dropped, because a method that cannot run on this photo is a result about the
     * method.
     *
     * <p>Blocking and heavy — several seconds on a full-resolution photo. Call it off the main
     * thread.
     */
    public static List<MethodResult> runAll(Context context, String uriString) {
        List<MethodResult> results = new ArrayList<>();
        if (!HardwareFeatureExtractor.isAvailable()) {
            Log.e(TAG, "OpenCV unavailable — no method can run.");
            return results;
        }

        Bitmap bitmap = HardwareFeatureExtractor.loadBitmap(
                context, uriString, HardwareFeatureExtractor.WORKING_MAX_DIM);
        if (bitmap == null) {
            Log.e(TAG, "Could not load " + uriString);
            return results;
        }

        Mat src = new Mat();
        Mat gray = new Mat();
        String inUse = selectedId(context);
        try {
            Utils.bitmapToMat(bitmap, src);
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            for (Method method : catalogue()) {
                results.add(run(method, gray, src, method.id.equals(inUse)));
            }
        } catch (Throwable t) {
            // UnsatisfiedLinkError is an Error, not an Exception: catching only Exception here
            // is what let an ABI mismatch kill a worker thread silently elsewhere in this app.
            Log.e(TAG, "Edge comparison failed outright", t);
        } finally {
            src.release();
            gray.release();
            bitmap.recycle();
        }
        return results;
    }

    /** Runs one method, measures it, and converts its map for display. */
    private static MethodResult run(Method method, Mat gray, Mat src, boolean inUse) {
        Mat edges = null;
        try {
            long startedAt = System.currentTimeMillis();
            // Unpinned: the lab looks at one photograph, so each method decides for itself.
            edges = method.operator.apply(gray, src, null, new boolean[1]);
            long elapsedMs = System.currentTimeMillis() - startedAt;

            double density = 100.0 * Core.countNonZero(edges) / (edges.rows() * edges.cols());
            int contours = countContours(edges);
            // findFeatureBoxes gets a clone: it is the detector's own code and is entitled to
            // consume what it is handed, while this map still has to be measured and drawn.
            Mat forBoxes = edges.clone();
            int boxes;
            try {
                boxes = HardwareFeatureExtractor
                        .findFeatureBoxes(forBoxes, edges.cols(), edges.rows()).size();
            } finally {
                forBoxes.release();
            }

            return new MethodResult(method.id, method.name, method.summary,
                    toDisplayBitmap(edges), density, contours, boxes, elapsedMs,
                    method.shipped, inUse, null);
        } catch (Throwable t) {
            Log.e(TAG, method.name + " failed", t);
            String reason = t.getMessage() == null ? t.getClass().getSimpleName() : t.getMessage();
            return new MethodResult(method.id, method.name, method.summary, null, 0, 0, 0, 0,
                    method.shipped, inUse, reason);
        } finally {
            if (edges != null) edges.release();
        }
    }

    private static int countContours(Mat edges) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat working = edges.clone();
        try {
            Imgproc.findContours(working, contours, hierarchy,
                    Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            return contours.size();
        } finally {
            for (MatOfPoint c : contours) c.release();
            hierarchy.release();
            working.release();
        }
    }

    /**
     * Shrinks a working-resolution edge map to something a phone can hold a dozen of.
     *
     * <p>INTER_AREA turns a one-pixel line into a grey one rather than dropping it, which is
     * what nearest-neighbour would do at this scale factor — a sparse map would look emptier
     * than it is, which is precisely the judgement this screen exists to support.
     */
    private static Bitmap toDisplayBitmap(Mat edges) {
        Mat scaled = new Mat();
        try {
            int longest = Math.max(edges.cols(), edges.rows());
            if (longest > DISPLAY_MAX_DIM) {
                double factor = (double) DISPLAY_MAX_DIM / longest;
                Imgproc.resize(edges, scaled, new Size(), factor, factor, Imgproc.INTER_AREA);
            } else {
                edges.copyTo(scaled);
            }
            // RGB_565 rather than ARGB_8888: these are opaque single-channel maps, so the
            // alpha byte and one colour byte per pixel would buy nothing across nine images.
            Bitmap bitmap = Bitmap.createBitmap(
                    scaled.cols(), scaled.rows(), Bitmap.Config.RGB_565);
            Utils.matToBitmap(scaled, bitmap);
            return bitmap;
        } finally {
            scaled.release();
        }
    }
}
