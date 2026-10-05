package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

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
 *   <li><b>The inspection's own input and output.</b> Every method sees the image after the
 *       chassis mask has painted out the background, as an inspection's edge stage does, and
 *       each card carries the boxes and crops an inspection would take from its map — padded,
 *       sourced and numbered the same way — so a method is judged on what the classifier
 *       would actually be handed.
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
 * ({@link #select}); the choice persists across launches and defaults to
 * {@link #ADAPTIVE_ID}, the entry that measures each photograph and picks one of three
 * strengths of Hybrid v6 for it — see {@link AdaptivePreprocessing}. The shipped stack keeps
 * its badge as the pipeline the detector was originally tuned with.
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
    /** Catalogue id of the stack the detector was originally tuned with; carries the badge. */
    public static final String SHIPPED_ID = "shipped-hybrid";
    /**
     * Catalogue id of the adaptive edge stage: {@link AdaptivePreprocessing}, which measures
     * the photograph and runs one of the {@link PreprocessingStrength} rungs.
     */
    public static final String ADAPTIVE_ID = "adaptive";
    /**
     * What runs until someone chooses. Adaptive rather than shipped: the shipped stack's own
     * per-image adaptation (bucketing Laplacian variance into a bilateral diameter) was found
     * firing backwards, and the ground-truth search behind v6 already showed the shipped stack
     * burying every fastener in extras (254 against v6's 13). The adaptive entry runs v6 for a
     * photograph it cannot fault and moves off it only on measured evidence.
     */
    public static final String DEFAULT_ID = ADAPTIVE_ID;

    /**
     * Longest edge of the edge-map thumbnails handed back for display.
     *
     * <p>A dozen maps at the 1280px working resolution would be about 60MB of bitmap in
     * native memory at once, which on Android 8+ does not raise OutOfMemoryError — it grows
     * the process until the kernel kills it. Metrics are still measured at full working
     * resolution; only the picture is shrunk.
     */
    static final int DISPLAY_MAX_DIM = 640;

    /**
     * Crops shown per card. The noisy operators find hundreds of candidates, and every
     * thumbnail is a region decode plus a bitmap; past a few dozen the strip stops being
     * readable anyway, and the box overlay above it still shows every one.
     */
    public static final int MAX_CROPS_SHOWN = 30;

    /** Longest edge of a crop thumbnail. Enough to tell a screw from a shadow. */
    static final int CROP_THUMB_DIM = 128;

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
        /**
         * What a method that decides something per photograph decided, or null. The adaptive
         * entry reports the rung it ran and the measurements behind it; a card that only
         * showed its counts would leave the operator matching numbers against the rung cards
         * to work out which one it was.
         */
        public final String note;
        /**
         * Every box an inspection would crop from this map, as fractions of the frame, in the
         * order the inspection numbers them: after the geometry filter, and with anything on
         * the calibration marker dropped, exactly as {@code performDynamicDetection} does.
         */
        public final List<RectF> boxes;
        /**
         * Thumbnails of the first {@link #MAX_CROPS_SHOWN} of {@link #boxes}, framed with the
         * inspection's own padding and cut from the same source it would cut them from. An
         * entry is null where the crop could not be read; the inspection skips those too.
         */
        public final List<Bitmap> crops;
        /**
         * True on the card the adaptive stage would run for this photo: the method that best
         * reproduces the chosen part's reference, or without a part, the image-quality
         * starting point. Set once every method has been scored.
         */
        public boolean picked;
        /**
         * How far this method's boxes sit from the chosen part's reference ({@link
         * SizePrior#agreement}, lower is better); NaN when no part was chosen or the boxes
         * could not be aligned onto it. The number the adaptive stage decides on.
         */
        public double referenceScore = Double.NaN;

        MethodResult(String id, String name, String summary, Bitmap edgeMap,
                     double edgeDensityPercent, int contourCount, int candidateBoxes,
                     long elapsedMs, boolean isShippedPipeline, boolean inUse, String failure) {
            this(id, name, summary, edgeMap, edgeDensityPercent, contourCount, candidateBoxes,
                    elapsedMs, isShippedPipeline, inUse, failure, null);
        }

        MethodResult(String id, String name, String summary, Bitmap edgeMap,
                     double edgeDensityPercent, int contourCount, int candidateBoxes,
                     long elapsedMs, boolean isShippedPipeline, boolean inUse, String failure,
                     String note) {
            this(id, name, summary, edgeMap, edgeDensityPercent, contourCount, candidateBoxes,
                    elapsedMs, isShippedPipeline, inUse, failure, note,
                    Collections.<RectF>emptyList(), Collections.<Bitmap>emptyList(), false);
        }

        MethodResult(String id, String name, String summary, Bitmap edgeMap,
                     double edgeDensityPercent, int contourCount, int candidateBoxes,
                     long elapsedMs, boolean isShippedPipeline, boolean inUse, String failure,
                     String note, List<RectF> boxes, List<Bitmap> crops,
                     boolean picked) {
            this.boxes = Collections.unmodifiableList(new ArrayList<>(boxes));
            this.crops = Collections.unmodifiableList(new ArrayList<>(crops));
            this.picked = picked;
            this.note = note;
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

        /** How many crops the card shows against how many the inspection would make. */
        public String cropsLine() {
            if (failed()) return "";
            if (boxes.isEmpty()) return "An inspection would crop nothing from this map.";
            if (crops.size() < boxes.size()) {
                return String.format(Locale.US,
                        "An inspection would crop and classify %d features; first %d shown:",
                        boxes.size(), crops.size());
            }
            return String.format(Locale.US, "An inspection would crop and classify %d %s:",
                    boxes.size(), boxes.size() == 1 ? "feature" : "features");
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
         * @param noteOut    one-element array; a method that decides something per image may
         *                   write one line saying what, for its card. Most leave it alone.
         * @return a fresh CV_8UC1 map of 0/255, which the caller releases
         */
        Mat apply(Mat gray, Mat srcRgba, Boolean forceGlare, boolean[] glareOut,
                  String[] noteOut);
    }

    /** Catalogue entry: what the method is called, what it does, and how to run it. */
    static final class Method {
        /** Stable id, persisted by {@link #select}; never renamed once shipped. */
        final String id;
        final String name;
        /** What the method is good at, in a sentence or two, for the screens that offer it. */
        final String goodAt;
        /** The key steps it runs, as a short list. */
        final String uses;
        final String summary;
        final boolean shipped;
        final Operator operator;

        Method(String id, String name, String goodAt, String uses, String summary,
               boolean shipped, Operator operator) {
            this.id = id;
            this.name = name;
            this.goodAt = goodAt;
            this.uses = uses;
            this.summary = summary;
            this.shipped = shipped;
            this.operator = operator;
        }
    }

    /**
     * The methods compared, in the order they are shown.
     *
     * <p>In version order, Hybrid v1 to v10, then the adaptive stage that chooses among them:
     * v1 is the stack the detector shipped with; v2 to v6 the tuned replacements for it, each
     * from a study run; v7 and v8 the light and strong versions of v6 the adaptive stage
     * reaches for on soft or noisy photographs; v9 and v10 the two Canny baselines the rest
     * are built up from. The ids are older than the names and are kept, because layouts and
     * preferences store them.
     *
     * <p>The bare operators that used to open the list — Canny at fixed thresholds, Sobel and
     * Scharr gradient magnitude, Laplacian of Gaussian, morphological gradient and adaptive
     * threshold — were removed: on screws_none.jpg, six fasteners, they found 13, 12, 121, 68,
     * 12 and 69 boxes, and none was ever a method worth running.
     */
    static List<Method> catalogue() {
        List<Method> methods = new ArrayList<>();

        methods.add(new Method(SHIPPED_ID, "Hybrid v1",
                "Not missing hardware: it never came in under the hand count on the test "
                        + "photos, faint and out-of-focus fasteners included. The cost is extra "
                        + "boxes on scratches and texture.",
                "Glare correction when the frame is blown out · CLAHE · bilateral (size "
                        + "set per photo) · median 5 + Gaussian 5 · Canny at Otsu, low 0.5x · "
                        + "dilate 5, close 7",
                "What the detector actually runs: black-hat glare correction when the frame is "
                        + "blown out, CLAHE, a noise-adaptive bilateral filter, median and "
                        + "Gaussian blurs, Otsu-scaled Canny, then dilate and close to bridge a "
                        + "fragmented outline into one contour. Called directly rather than "
                        + "reproduced, so this row cannot drift away from the shipped code.",
                true,
                (gray, srcRgba, forceGlare, glareOut, noteOut) -> {
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
        methods.add(new Method("hybrid-v2", "Hybrid v2",
                "Keeping the same count when the lighting changes: the steadiest under "
                        + "dimming, brightening, noise and JPEG. Exact on 12 of 26 test photos, "
                        + "with the most misses (5).",
                "CLAHE · bilateral 9 · median 5 + Gaussian 5 · Canny at 1.1x Otsu, low "
                        + "0.4x · dilate 5, close 7",
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
                (gray, srcRgba, forceGlare, glareOut, noteOut) -> hybridVariant(srcRgba, forceGlare, glareOut, 9, 5, 5, 1.1, 0.4, false, 5, 7)));

        methods.add(new Method("hybrid-v3", "Hybrid v3",
                "A steady all-rounder, and the cheapest hybrid. Exact on 12 of 26 test "
                        + "photos, with 3 misses.",
                "CLAHE · bilateral 5 · median 5 + Gaussian 7 · Canny at 1.2x Otsu, low "
                        + "0.4x · dilate 5, close 7",
                "Bilateral 5, a wider Gaussian at 7, Canny at 1.2x Otsu with the same 0.4 "
                        + "low ratio. Best all-rounder in the search — it took the top combined "
                        + "rank across all three robustness tests without winning any of them "
                        + "outright: near-perfect under photographic variation, on a broad "
                        + "parameter plateau, group stability 0.18, and the cheapest. Against "
                        + "the full ground truth: 12 of 26 exact, 3 misses, 33 extras.",
                false,
                (gray, srcRgba, forceGlare, glareOut, noteOut) -> hybridVariant(srcRgba, forceGlare, glareOut, 5, 5, 7, 1.2, 0.4, false, 5, 7)));

        methods.add(new Method("hybrid-v4", "Hybrid v4",
                "Giving the same answer on two photos of the same part: the best "
                        + "stability, so the fewest false FAILs. Exact on 16 of 26 test photos.",
                "CLAHE · bilateral 7 · median 5 + Gaussian 5 · Canny at Otsu with the "
                        + "true (L2) gradient, low 0.4x · dilate 3, close 9",
                "Bilateral 7, the 0.4 low ratio, a true Euclidean gradient magnitude rather "
                        + "than the |Gx|+|Gy| approximation, and lighter morphology — a 3px "
                        + "dilation closed with a 9px ellipse. Best group stability of any "
                        + "configuration tested (0.13), which is the number that predicts false "
                        + "FAILs. Against the full ground truth it was far ahead of v2, v3 and "
                        + "v5 all along: 16 of 26 exact, 3 misses, 15 extras.",
                false,
                (gray, srcRgba, forceGlare, glareOut, noteOut) -> hybridVariant(srcRgba, forceGlare, glareOut, 7, 5, 5, 1.0, 0.4, true, 3, 9)));

        methods.add(new Method("hybrid-v5", "Hybrid v5",
                "Ignoring plate texture, by keeping only edges beside a dark region. No "
                        + "better than v3 overall (12 of 26), at twice the cost.",
                "Hybrid v3's steps · plus an adaptive-threshold mask (61 px "
                        + "neighbourhood, grown 5 px) over the edges",
                "v3's pipeline with an adaptive-threshold mask over the Canny map: only edges "
                        + "lying on or beside a locally dark region are kept, which is what a "
                        + "fastener is against a machined plate. The answer to whether adaptive "
                        + "thresholding helps the hybrids on the proxy tests — it does, as a "
                        + "mask and not as a detector. Against the full ground truth it does "
                        + "not: 12 of 26 exact, no better than v3, at twice the cost. Kept so "
                        + "the negative result can be seen rather than taken on trust.",
                false,
                (gray, srcRgba, forceGlare, glareOut, noteOut) ->
                        hybridVariant(srcRgba, forceGlare, glareOut, 5, 5, 7, 1.2, 0.4, false, 5, 7, 61, 2, 5,
                                CLAHE_CLIP_LIMIT)));

        methods.add(new Method(PreprocessingStrength.MEDIUM.id, "Hybrid v6",
                "Accuracy: the best overall, exact on 20 of 26 test photos and on every "
                        + "stud and screw shot. The usual starting point when no count is given.",
                "CLAHE · bilateral 5 · median 5 + Gaussian 5 · Canny at 1.1x Otsu with "
                        + "the L2 gradient, low 0.4x · dilate 5, close 9",
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
                (gray, srcRgba, forceGlare, glareOut, noteOut) ->
                        outlineEdges(srcRgba, forceGlare, glareOut)));

        // The adaptive stage's other two rungs, v7 and v8. v6 is its middle rung and is
        // listed once, above, under its own id: a second card with identical output
        // would read as a second method. See PreprocessingStrength for why the rungs move
        // CLAHE's clip limit and not the filter apertures, and
        // python/adaptive_preprocessing_study.py for every number below.
        methods.add(new Method(PreprocessingStrength.LIGHT.id, "Hybrid v7",
                "Soft or blurred photos and small hardware, where edges are faint. On "
                        + "sharp photos it boxes far too much.",
                "Hybrid v6 with double the CLAHE contrast (2.0) and a lighter Gaussian "
                        + "(3)",
                "Hybrid v6 with twice the CLAHE contrast boost and a 3px Gaussian instead of "
                        + "5px, so a faint edge survives to Canny. The rung the adaptive stage "
                        + "runs when the photograph's edges are weak (99th-percentile Laplacian "
                        + "response under 20): on the bundled set blurred 9px it scores 24 "
                        + "against v6's 41. On anything sharp it is catastrophic — 25 misses "
                        + "and 205 extras on the clean set. Also run when a sharp photo's "
                        + "hardware is small (the medium rung's boxes under 25px on the short "
                        + "side), where v6's closing welds neighbours together: corner.jpg goes "
                        + "from 6 to 9 of its 10.",
                false,
                (gray, srcRgba, forceGlare, glareOut, noteOut) ->
                        strengthEdges(srcRgba, forceGlare, glareOut, PreprocessingStrength.LIGHT)));

        methods.add(new Method(PreprocessingStrength.STRONG.id, "Hybrid v8",
                "Noisy or grainy photos and textured plates, by not amplifying the grain. "
                        + "Can lose low-contrast heads on clean photos.",
                "Hybrid v6 with half the CLAHE contrast (0.5) and Canny at 1.0x Otsu",
                "Hybrid v6 with half the CLAHE contrast boost and Canny's high bound back at "
                        + "1.0x Otsu. CLAHE amplifies grain in flat regions as readily as it "
                        + "lifts faint edges, and under noise the grain wins; this rung stops "
                        + "amplifying it. Chosen when the robust noise estimate reaches 12 — "
                        + "the textured plates, or sigma 7 of added noise. At sigma 10 it scores "
                        + "20 against v6's 31; on clean photographs it costs a few low-contrast "
                        + "heads (28 against 19). Same apertures as v6, same cost. A heavier "
                        + "bilateral was measured for this role and erased small hardware.",
                false,
                (gray, srcRgba, forceGlare, glareOut, noteOut) ->
                        strengthEdges(srcRgba, forceGlare, glareOut, PreprocessingStrength.STRONG)));

        // The two Canny baselines, v9 and v10: each is the shipped stack with most of it
        // taken away, so what the hybrids above add can be read against them.
        methods.add(new Method("clahe-bilateral-canny", "Hybrid v9",
                "A quick baseline: what evening out the contrast achieves on its own. "
                        + "With no blur or joining step, outlines can break into pieces.",
                "CLAHE · bilateral 5 · Canny at Otsu, low 0.5x · no glare correction, "
                        + "blur or morphology",
                "The shipped stack with the glare branch, the two blurs and the morphology all "
                        + "removed: local contrast equalisation, an edge-preserving denoise, "
                        + "then Otsu-scaled Canny. Isolates what the contrast work contributes, "
                        + "so Hybrid v1 can be read as the cost of everything else.",
                false,
                (gray, srcRgba, forceGlare, glareOut, noteOut) -> {
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

        methods.add(new Method("canny-otsu", "Hybrid v10",
                "Simplicity: fine on clean, evenly lit photos. Faint or low-contrast "
                        + "hardware breaks up or disappears.",
                "Gaussian 5 · Canny at Otsu, low 0.5x · nothing else",
                "Canny with its two thresholds derived per image from Otsu's method rather "
                        + "than hard-coded, after a 5x5 Gaussian, and no other preprocessing. "
                        + "Deriving the thresholds is the single change that lets one setting "
                        + "survive a change of lighting; every other method adds to it.",
                false,
                (gray, srcRgba, forceGlare, glareOut, noteOut) -> {
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

        methods.add(new Method(ADAPTIVE_ID, "Adaptive (chooses a method per photograph)",
                "Choosing for you: on each inspection photo it keeps the method that best "
                        + "reproduces the part's labelled reference.",
                "Every method from Hybrid v1 to v10, scored against the reference layout",
                "On every inspection photograph, runs every other method on this screen and "
                        + "scores each one's boxes against the part's labelled reference: a "
                        + "box the wrong size, a slot nothing matched and a box invented on "
                        + "the part all count against it. The method that best reproduces "
                        + "what the operator labelled runs; the reference's own method is kept "
                        + "unless another beats it clearly. Without a reference (reference "
                        + "setup itself) it starts from the photo's noise and contrast, and the "
                        + "operator can re-run with any method. Pick a part at the top of this "
                        + "screen to see the real choice on this card.",
                false,
                (gray, srcRgba, forceGlare, glareOut, noteOut) -> {
                    AdaptivePreprocessing.Choice[] choice = new AdaptivePreprocessing.Choice[1];
                    HardwareFeatureExtractor.Edges edges =
                            AdaptivePreprocessing.compute(srcRgba, forceGlare, null, choice, null);
                    try {
                        glareOut[0] = edges.glareCorrectionApplied;
                        if (choice[0] != null) noteOut[0] = choice[0].describe();
                        return edges.edgesClosed.clone();
                    } finally {
                        edges.release();
                    }
                }));

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
        // v6 is the adaptive stage's middle rung, and this is the one place its parameters
        // are written down — so the ghost outline, the Edge Lab card and the rung cannot
        // drift apart.
        return strengthEdges(srcRgba, forceGlare, glareOut, PreprocessingStrength.MEDIUM);
    }

    /**
     * One rung of the adaptive stage: Hybrid v6's pipeline — bilateral 5, median 5, Canny
     * low ratio 0.4 with the L2 gradient, dilate 5, close 9 — with the rung's CLAHE clip,
     * Gaussian aperture and Otsu multiplier. {@link PreprocessingStrength#MEDIUM} reproduces
     * v6 exactly.
     *
     * @return a fresh CV_8UC1 map of 0/255, which the caller releases
     */
    static Mat strengthEdges(Mat srcRgba, Boolean forceGlare, boolean[] glareOut,
                             PreprocessingStrength rung) {
        return hybridVariant(srcRgba, forceGlare, glareOut, 5, 5, rung.gaussK,
                rung.otsuHighScale, 0.4, true, 5, 9, 0, 0, 0, rung.claheClip);
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
                otsuHighScale, lowRatio, l2Gradient, dilateK, closeK, 0, 0, 0, CLAHE_CLIP_LIMIT);
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
     * @param claheClip CLAHE clip limit. Production's 1.0 for every tuned variant; the
     *                  adaptive rungs are the only callers that move it, and it is the knob
     *                  that measured as the lever for noise and blur alike
     */
    private static Mat hybridVariant(Mat srcRgba, Boolean forceGlare, boolean[] glareOut,
                                     int bilateralD, int medianK, int gaussK,
                                     double otsuHighScale, double lowRatio, boolean l2Gradient,
                                     int dilateK, int closeK,
                                     int maskBlock, int maskC, int maskGrow, double claheClip) {
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

            CLAHE clahe = Imgproc.createCLAHE(claheClip, new Size(16, 16));
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

    /** Id of the method inspections currently run; {@link #DEFAULT_ID} until one is chosen. */
    public static String selectedId(Context context) {
        String id = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString(KEY_SELECTED, DEFAULT_ID);
        // An id from a build that had a method this one does not is not an error worth a
        // crash, and silently running an arbitrary method would be worse: fall back to the
        // default.
        return byId(id) == null ? DEFAULT_ID : id;
    }

    /** Display name of the method inspections currently run. */
    public static String selectedName(Context context) {
        Method m = byId(selectedId(context));
        return m == null ? nameOf(DEFAULT_ID) : m.name;
    }

    /** Display name for a catalogue id, or the id itself when it is not catalogued. */
    public static String nameOf(String id) {
        Method m = byId(id);
        return m == null ? String.valueOf(id) : m.name;
    }

    /** What a method is good at, in a sentence or two; empty when it is not catalogued. */
    public static String goodAtOf(String id) {
        Method m = byId(id);
        return m == null ? "" : m.goodAt;
    }

    /** The key steps a method runs, as a short list; empty when it is not catalogued. */
    public static String usesOf(String id) {
        Method m = byId(id);
        return m == null ? "" : m.uses;
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

    /** True when the chosen method is the shipped stack. */
    public static boolean shippedSelected(Context context) {
        return SHIPPED_ID.equals(selectedId(context));
    }

    /** True when nothing has been overridden and the default runs. */
    public static boolean defaultSelected(Context context) {
        return DEFAULT_ID.equals(selectedId(context));
    }

    /** True when the chosen method is the adaptive stage, which picks a rung per photograph. */
    public static boolean adaptiveSelected(Context context) {
        return ADAPTIVE_ID.equals(selectedId(context));
    }

    static Method byId(String id) {
        if (id == null) return null;
        for (Method m : catalogue()) if (m.id.equals(id)) return m;
        return null;
    }

    /** The catalogue entry inspections run; never null. */
    static Method selected(Context context) {
        Method m = byId(selectedId(context));
        return m != null ? m : byId(DEFAULT_ID);
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
            map = method.operator.apply(gray, src, forceGlare, glare, new String[1]);
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
     * Everything one lab run produced: a result per method, what the photograph measured as,
     * and — when a part was chosen — how every method scored against that part's reference
     * and which one the adaptive stage would therefore run.
     */
    public static final class Comparison {
        public final List<MethodResult> results;
        /** Null when the image could not be read. */
        public final ImageQuality quality;
        /** Fraction of the frame the chassis mask excluded before any method saw it. */
        public final double backgroundFraction;
        /** Whether a calibration marker was found, so boxes on it were dropped as inspections do. */
        public final boolean markerFound;
        /** The method inspections currently run. */
        public final String selectedId;
        /** The part scored against, or null. */
        public final String partId;
        /** Why no part could be scored against, when one was asked for; else null. */
        public final String partProblem;
        /** What the prior is built from, for the header; null without one. */
        public final String priorDescription;
        /** Score per competing method; empty without a part. */
        public final Map<String, Double> scores;
        /** What runs unless beaten: the reference's method, or the image-quality start. */
        public final String incumbentId;
        /** What the adaptive stage would run on this photo. */
        public final String chosenId;

        Comparison(List<MethodResult> results, ImageQuality quality, double backgroundFraction,
                   boolean markerFound, String selectedId, String partId, String partProblem,
                   String priorDescription, Map<String, Double> scores, String incumbentId,
                   String chosenId) {
            this.results = results;
            this.quality = quality;
            this.backgroundFraction = backgroundFraction;
            this.markerFound = markerFound;
            this.selectedId = selectedId;
            this.partId = partId;
            this.partProblem = partProblem;
            this.priorDescription = priorDescription;
            this.scores = scores;
            this.incumbentId = incumbentId;
            this.chosenId = chosenId;
        }

        static Comparison empty(String selectedId, String partId) {
            return new Comparison(new ArrayList<>(), null, 0, false, selectedId, partId, null,
                    null, new LinkedHashMap<>(), null, null);
        }

        public boolean isEmpty() {
            return results.isEmpty();
        }

        /** The header's explanation of the choice. See {@link EdgeMethodLab#verdict}. */
        public List<String> verdict() {
            return EdgeMethodLab.verdict(quality, selectedId, partId, partProblem,
                    priorDescription, scores, incumbentId, chosenId);
        }

        /**
         * Every bitmap this run holds, for a caller that has to free them. A crop found by
         * several methods is one bitmap shared between their cards, so it appears once.
         */
        public List<Bitmap> bitmaps() {
            List<Bitmap> out = new ArrayList<>();
            java.util.Set<Bitmap> seen =
                    java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
            for (MethodResult r : results) {
                if (r.edgeMap != null && seen.add(r.edgeMap)) out.add(r.edgeMap);
                for (Bitmap b : r.crops) if (b != null && seen.add(b)) out.add(b);
            }
            return out;
        }
    }

    /**
     * Parts with a labelled reference layout on this device: the parts the lab can score
     * methods against.
     */
    public static List<String> partsWithReference(Context context) {
        return FeatureLayout.savedPartIds(context);
    }

    /**
     * The prior an inspection of {@code partId} would score methods against: its labelled
     * layout, critical slots only, exactly as the results screen builds it, except that the
     * reference-URI guard is skipped, since the lab's photograph is by definition not the
     * reference. Null when the part cannot be scored against.
     */
    static SizePrior priorForPart(Context context, String partId) {
        FeatureLayout layout = FeatureLayout.load(context, partId);
        if (layout == null) return null;
        FeatureLayout critical = layout.retaining(s -> FeatureTypeRegistry.severityOf(
                context, s.type) == FeatureTypeRegistry.Severity.CRITICAL);
        return SizePrior.from(critical.isEmpty() ? null : critical);
    }

    private static String priorProblem(Context context, String partId) {
        FeatureLayout layout = FeatureLayout.load(context, partId);
        if (layout == null) return "it has no labelled reference layout";
        return "its reference has too few critical slots with a usable box to score against ("
                + layout.slots().size() + " slots; at least 4 critical ones of 30 px or more "
                + "are needed)";
    }

    /**
     * Runs every catalogued method over one image, the way an inspection would.
     *
     * <p>The photograph goes through the steps {@code extractFeatures} takes before its edge
     * stage (calibration marker detection on the raw pixels, then the chassis mask filling
     * the background with the plate's grey), so each card's map, box count and crops are what
     * that method would produce if it were chosen. Boxes on the marker are dropped as the
     * inspection drops them, and each crop is padded by
     * {@link HardwareFeatureExtractor#cropPaddingFor} and cut from the source photo when that
     * has more detail than the working copy: the pixels the encoder would be handed.
     *
     * <p>With a part, every method's boxes are also scored against that part's labelled
     * reference, the same {@link SizePrior#agreement} an inspection decides on, and the
     * adaptive card runs exactly as an inspection of that part would, so the lab shows the
     * real per-photograph choice rather than a fallback.
     *
     * <p>Returns an empty comparison when OpenCV is unavailable or the image cannot be read. A
     * method that throws comes back as a {@link MethodResult} carrying its failure rather than
     * being dropped, because a method that cannot run on this photo is a result about the
     * method.
     *
     * <p>Blocking and heavy: several seconds on a full-resolution photo. Call it off the main
     * thread.
     *
     * @param partId a part to score against, or null
     */
    public static Comparison runAll(Context context, String uriString, String partId) {
        List<MethodResult> results = new ArrayList<>();
        String inUse = selectedId(context);
        if (!HardwareFeatureExtractor.isAvailable()) {
            Log.e(TAG, "OpenCV unavailable — no method can run.");
            return Comparison.empty(inUse, partId);
        }

        Bitmap bitmap = HardwareFeatureExtractor.loadBitmap(
                context, uriString, HardwareFeatureExtractor.WORKING_MAX_DIM);
        if (bitmap == null) {
            Log.e(TAG, "Could not load " + uriString);
            return Comparison.empty(inUse, partId);
        }

        final SizePrior prior = partId == null ? null : priorForPart(context, partId);
        String partProblem = partId != null && prior == null ? priorProblem(context, partId) : null;

        Mat src = new Mat();
        Mat gray = new Mat();
        ImageQuality quality = null;
        double backgroundFraction = 0;
        ScaleCalibration calibration = null;
        CropSource crops = null;
        Map<String, Double> scores = new LinkedHashMap<>();
        String incumbent = null;
        String chosen = null;
        try {
            Utils.bitmapToMat(bitmap, src);
            // Same order as extractFeatures: the marker is found on real pixels, then what is
            // not the chassis is painted out before any edge operator sees it.
            calibration = ScaleCalibration.detect(src);
            if (HardwareFeatureExtractor.MASK_BACKGROUND) {
                ChassisMask.Result bg = ChassisMask.of(src);
                try {
                    ChassisMask.fillWithPlate(src, bg);
                    backgroundFraction = bg.fraction;
                } finally {
                    bg.release();
                }
            }
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            quality = ImageQuality.assess(gray);

            List<String> ballot = AdaptivePreprocessing.ballot();
            incumbent = quality.suggest().id;
            if (prior != null && prior.referenceMethodId() != null
                    && ballot.contains(prior.referenceMethodId())) {
                incumbent = prior.referenceMethodId();
            }

            crops = new CropSource(SourceCropper.open(
                    context, uriString, bitmap.getWidth(), bitmap.getHeight()), src);
            for (Method method : catalogue()) {
                Method toRun = method;
                if (ADAPTIVE_ID.equals(method.id) && prior != null) {
                    // What an inspection of this part would run: every method scored against
                    // its reference. Not the catalogue entry, which has no part to score
                    // against and so could only show the starting point.
                    toRun = new Method(method.id, method.name, method.goodAt, method.uses,
                            method.summary, false,
                            (g, s, fg, go, no) -> {
                                AdaptivePreprocessing.Choice[] choice =
                                        new AdaptivePreprocessing.Choice[1];
                                HardwareFeatureExtractor.Edges e = AdaptivePreprocessing
                                        .compute(s, fg, prior, choice, null);
                                try {
                                    go[0] = e.glareCorrectionApplied;
                                    if (choice[0] != null) no[0] = choice[0].describe();
                                    return e.edgesClosed.clone();
                                } finally {
                                    e.release();
                                }
                            });
                }
                MethodResult r = run(toRun, gray, src, method.id.equals(inUse), calibration,
                        crops, prior);
                results.add(r);
                if (prior != null && ballot.contains(r.id)) scores.put(r.id, r.referenceScore);
            }
            chosen = prior != null
                    ? AdaptivePreprocessing.choose(incumbent, scores,
                            AdaptivePreprocessing.AGREEMENT_MARGIN)
                    : incumbent;
            for (MethodResult r : results) r.picked = r.id.equals(chosen);
        } catch (Throwable t) {
            // UnsatisfiedLinkError is an Error, not an Exception: catching only Exception here
            // is what let an ABI mismatch kill a worker thread silently elsewhere in this app.
            Log.e(TAG, "Edge comparison failed outright", t);
        } finally {
            if (crops != null) crops.close();
            src.release();
            gray.release();
            bitmap.recycle();
        }
        return new Comparison(results, quality, backgroundFraction, calibration != null, inUse,
                partId, partProblem, prior == null ? null : prior.describe(), scores, incumbent,
                chosen);
    }

    /**
     * What the adaptive stage would do with this photograph and why, as plain lines for the
     * lab's header. Pure, so the wording of the decision is testable without OpenCV.
     *
     * @param quality          the photograph's measurements
     * @param selectedId       the method inspections currently run
     * @param partId           the part scored against, or null
     * @param partProblem      why that part could not be scored against, or null
     * @param priorDescription what the part's prior is built from, or null
     * @param scores           each competing method's score against the part; empty without
     * @param incumbentId      what runs unless beaten
     * @param chosenId         what the adaptive stage would run
     */
    public static List<String> verdict(ImageQuality quality, String selectedId, String partId,
                                       String partProblem, String priorDescription,
                                       Map<String, Double> scores, String incumbentId,
                                       String chosenId) {
        List<String> lines = new ArrayList<>();
        if (quality == null || chosenId == null) return lines;
        if (scores != null && !scores.isEmpty()) {
            lines.add("Every method scored against " + partId + "'s labelled reference ("
                    + priorDescription + "). Lower is closer to what the operator labelled: a "
                    + "wrong-sized box, a missed slot and an invented box on the part all "
                    + "count against a method.");
            List<Map.Entry<String, Double>> ranked = new ArrayList<>(scores.entrySet());
            ranked.sort((a, b) -> Double.compare(orInf(a.getValue()), orInf(b.getValue())));
            int shown = 0;
            for (Map.Entry<String, Double> e : ranked) {
                if (shown++ == 5) break;
                double v = orInf(e.getValue());
                lines.add(String.format(Locale.US, "   %s   %s%s",
                        Double.isInfinite(v) ? "  —  " : String.format(Locale.US, "%.3f", v),
                        nameOf(e.getKey()), e.getKey().equals(incumbentId)
                                ? "   (reference's method)" : ""));
            }
            if (chosenId.equals(incumbentId)) {
                lines.add("The reference's method stands: nothing beat it by more than "
                        + fmt(AdaptivePreprocessing.AGREEMENT_MARGIN) + ".");
            } else {
                lines.add(nameOf(chosenId) + " reproduces the reference better than "
                        + (incumbentId == null ? "the starting point" : nameOf(incumbentId))
                        + " by more than " + fmt(AdaptivePreprocessing.AGREEMENT_MARGIN)
                        + ", so it runs.");
            }
        } else {
            if (partProblem != null) {
                lines.add("Cannot score against " + partId + ": " + partProblem + ".");
            }
            lines.add("No reference to score against, so this is only the starting point, from "
                    + "the photo's statistics (" + quality.describe() + "). Pick a part above to "
                    + "see what an inspection of it would actually run: every method scored "
                    + "against the operator's labelled reference.");
        }
        if (!ADAPTIVE_ID.equals(selectedId)) {
            lines.add("Note: inspections are set to " + nameOf(selectedId) + ", which does not "
                    + "adapt. Choose USE FOR INSPECTIONS on the Adaptive card for any of this "
                    + "to happen.");
        }
        return lines;
    }

    private static double orInf(Double v) {
        return v == null || Double.isNaN(v) ? Double.POSITIVE_INFINITY : v;
    }

    private static String fmt(double v) {
        return v == Math.rint(v) ? String.valueOf((long) v) : String.valueOf(v);
    }

    /**
     * Cuts crop thumbnails the way the inspection cuts crops, once per distinct box.
     *
     * <p>Many methods find many of the same boxes (the three rungs especially), and the region
     * decode is the expensive part, so each distinct rectangle is decoded once and its
     * thumbnail shared between the cards that found it.
     */
    private static final class CropSource {
        private final SourceCropper cropper;
        private final Mat src;
        private final Map<String, Bitmap> cache = new HashMap<>();

        CropSource(SourceCropper cropper, Mat src) {
            this.cropper = cropper;
            this.src = src;
        }

        Bitmap thumbnail(Rect r) {
            String key = r.x + "," + r.y + "," + r.width + "," + r.height;
            if (cache.containsKey(key)) return cache.get(key);
            Bitmap full = HardwareFeatureExtractor.cropFeature(cropper, src, r);
            Bitmap thumb = full == null ? null : shrink(full, CROP_THUMB_DIM);
            if (full != null && thumb != full) full.recycle();
            cache.put(key, thumb);
            return thumb;
        }

        void close() {
            if (cropper != null) cropper.close();
        }

        private static Bitmap shrink(Bitmap full, int maxDim) {
            int longest = Math.max(full.getWidth(), full.getHeight());
            if (longest <= maxDim) return full;
            float f = (float) maxDim / longest;
            return Bitmap.createScaledBitmap(full, Math.max(1, Math.round(full.getWidth() * f)),
                    Math.max(1, Math.round(full.getHeight() * f)), true);
        }
    }

    /** Runs one method, measures it, cuts its crops and converts its map for display. */
    private static MethodResult run(Method method, Mat gray, Mat src, boolean inUse,
                                    ScaleCalibration calibration, CropSource cropSource,
                                    SizePrior prior) {
        Mat edges = null;
        try {
            long startedAt = System.currentTimeMillis();
            // Unpinned: the lab looks at one photograph, so each method decides for itself.
            String[] note = new String[1];
            edges = method.operator.apply(gray, src, null, new boolean[1], note);
            long elapsedMs = System.currentTimeMillis() - startedAt;

            double density = 100.0 * Core.countNonZero(edges) / (edges.rows() * edges.cols());
            int contours = countContours(edges);
            // findFeatureBoxes gets a clone: it is the detector's own code and is entitled to
            // consume what it is handed, while this map still has to be measured and drawn.
            Mat forBoxes = edges.clone();
            List<Rect> found;
            try {
                found = HardwareFeatureExtractor
                        .findFeatureBoxes(forBoxes, edges.cols(), edges.rows());
            } finally {
                forBoxes.release();
            }

            float w = edges.cols(), h = edges.rows();
            List<RectF> boxes = new ArrayList<>();
            List<Bitmap> crops = new ArrayList<>();
            for (Rect r : found) {
                // The marker is scenery, not hardware; the inspection never crops it.
                if (calibration != null && calibration.overlapsMarker(r)) continue;
                boxes.add(new RectF(r.x / w, r.y / h, (r.x + r.width) / w, (r.y + r.height) / h));
                if (crops.size() < MAX_CROPS_SHOWN) crops.add(cropSource.thumbnail(r));
            }

            MethodResult result = new MethodResult(method.id, method.name, method.summary,
                    toDisplayBitmap(edges), density, contours, boxes.size(), elapsedMs,
                    method.shipped, inUse, null, note[0], boxes, crops, false);
            if (prior != null) {
                // Every box, marker included, exactly as AdaptivePreprocessing scores them.
                List<RectF> all = new ArrayList<>();
                for (Rect r : found) {
                    all.add(new RectF(r.x / w, r.y / h, (r.x + r.width) / w,
                            (r.y + r.height) / h));
                }
                result.referenceScore = prior.agreement(all);
            }
            return result;
        } catch (Throwable t) {
            Log.e(TAG, method.name + " failed", t);
            String reason = t.getMessage() == null ? t.getClass().getSimpleName() : t.getMessage();
            return new MethodResult(method.id, method.name, method.summary, null, 0, 0, 0, 0,
                    method.shipped, inUse, reason, null, Collections.<RectF>emptyList(),
                    Collections.<Bitmap>emptyList(), false);
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
