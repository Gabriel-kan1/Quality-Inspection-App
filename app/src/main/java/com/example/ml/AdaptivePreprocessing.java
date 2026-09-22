package com.example.ml;

import android.graphics.RectF;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The edge stage that chooses its own preprocessing strength, per photograph.
 *
 * <p>Two sources of evidence, in order:
 * <ol>
 *   <li><b>The photograph itself.</b> {@link ImageQuality} measures how noisy it is and how
 *       much edge it has, and names the rung that evidence supports: strong for grain and
 *       dense texture, light for a soft image whose faint edges a harder stage would lose,
 *       medium — Hybrid v6 exactly — for everything else. This is always computed and is
 *       the answer when nothing else is known, e.g. during reference setup.</li>
 *   <li><b>The part's expected sizes.</b> When a labelled reference layout exists,
 *       {@link SizePrior} knows how big each critical feature should be — from the reference
 *       box, refined by every inspection filed since. Each eligible rung is run, its boxes
 *       aligned onto the layout, and the rung whose boxes come closest to those sizes wins,
 *       provided it beats the image-quality rung by {@link #AGREEMENT_MARGIN}. A shadow merged
 *       into a box, or a fastener half-cropped, is a size error before it is anything else,
 *       and this is the one test that sees it.</li>
 * </ol>
 *
 * <p>The margin matters. Size jitter between two genuine detections of the same hardware is a
 * median 0.058 in the same units ({@code python/size_ratio_study.py}), so a smaller advantage
 * is noise; and at 0.03 the rule made bad overrides — a light rung winning on size while
 * over-detecting everywhere else, which the size score cannot see. At 0.08 it never chose
 * worse than the image-quality rung over the same-scene pairs and improved count accuracy in
 * every degraded condition ({@code python/adaptive_preprocessing_study.py}, section 3).
 *
 * <p>Cost: every rung shares the same bilateral and median apertures, so three of them cost
 * three times one — about 60ms of edge work on a desktop plus three geometry passes and three
 * alignments, against several seconds of embedding for the crops that follow.
 */
public final class AdaptivePreprocessing {

    private static final String TAG = "AdaptivePreprocessing";

    /**
     * How much lower (in {@code |ln ratio|} units) another rung's size deviation must be
     * before it displaces the rung the image quality chose.
     */
    public static final double AGREEMENT_MARGIN = 0.08;

    private AdaptivePreprocessing() {}

    /** What ran and why, for the results banner and the log. */
    public static final class Choice {
        public final PreprocessingStrength rung;
        public final ImageQuality quality;
        /** The rung the image quality alone suggested. */
        public final PreprocessingStrength suggested;
        /** Size agreement per rung tried, NaN where the rung could not be aligned; empty when no prior. */
        public final Map<PreprocessingStrength, Double> agreement;
        /** The prior the rungs were scored against, or null. */
        public final SizePrior sizePrior;

        Choice(PreprocessingStrength rung, ImageQuality quality, PreprocessingStrength suggested,
               Map<PreprocessingStrength, Double> agreement, SizePrior sizePrior) {
            this.rung = rung;
            this.quality = quality;
            this.suggested = suggested;
            this.agreement = Collections.unmodifiableMap(new LinkedHashMap<>(agreement));
            this.sizePrior = sizePrior;
        }

        /** Whether the size test overrode the image-quality rung. */
        public boolean overridden() {
            return rung != suggested;
        }

        /** One sentence naming the rung and the evidence, for a banner. */
        public String describe() {
            StringBuilder sb = new StringBuilder();
            sb.append(rung.label.toUpperCase(Locale.US)).append(" preprocessing — ")
                    .append(quality.describe());
            if (agreement.isEmpty()) {
                sb.append(" (no reference sizes to test against)");
                return sb.toString();
            }
            sb.append("; ").append(suggested.label).append(" by image quality");
            if (overridden()) {
                sb.append(", overridden by size agreement");
            } else {
                sb.append(", confirmed by size agreement");
            }
            sb.append(" (");
            boolean first = true;
            for (Map.Entry<PreprocessingStrength, Double> e : agreement.entrySet()) {
                if (!first) sb.append(", ");
                first = false;
                sb.append(e.getKey().label).append(' ');
                sb.append(Double.isNaN(e.getValue()) ? "unaligned"
                        : String.format(Locale.US, "%.2f", e.getValue()));
            }
            sb.append("; ").append(sizePrior.describe()).append(')');
            return sb.toString();
        }
    }

    /**
     * The decision, separated from the OpenCV work so it can be tested: the suggested rung
     * stands unless some other rung's deviation is lower by more than {@code margin}. A rung
     * that could not be aligned (NaN) is never chosen. If the suggested rung itself could not
     * be aligned, the best aligned rung wins outright — a rung whose boxes cannot be registered
     * onto the part found something other than the part.
     */
    static PreprocessingStrength choose(PreprocessingStrength suggested,
                                        Map<PreprocessingStrength, Double> scores,
                                        double margin) {
        PreprocessingStrength best = null;
        double bestScore = Double.POSITIVE_INFINITY;
        for (Map.Entry<PreprocessingStrength, Double> e : scores.entrySet()) {
            double v = e.getValue() == null ? Double.NaN : e.getValue();
            if (Double.isNaN(v)) continue;
            if (v < bestScore) {
                bestScore = v;
                best = e.getKey();
            }
        }
        if (best == null) return suggested;
        Double own = scores.get(suggested);
        if (own == null || Double.isNaN(own)) return best;
        return bestScore < own - margin ? best : suggested;
    }

    /**
     * Runs the adaptive edge stage.
     *
     * @param src        working-resolution RGBA image; not modified
     * @param forceGlare pins the black-hat glare branch, as for every other method
     * @param sizePrior  expected sizes to arbitrate with, or null to decide on image quality
     *                   alone
     * @param choiceOut  one-element array that receives what was chosen and why; may be null
     * @return the maps, owned by the caller
     */
    static HardwareFeatureExtractor.Edges compute(Mat src, Boolean forceGlare,
                                                  SizePrior sizePrior, Choice[] choiceOut) {
        Mat gray = new Mat();
        Map<PreprocessingStrength, Mat> maps = new LinkedHashMap<>();
        boolean ok = false;
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            ImageQuality quality = ImageQuality.assess(gray);
            PreprocessingStrength suggested = quality.suggest();

            List<PreprocessingStrength> candidates = sizePrior == null
                    ? Collections.singletonList(suggested) : quality.candidates();
            Map<PreprocessingStrength, Double> scores = new LinkedHashMap<>();
            boolean glare = false;
            float w = src.width(), h = src.height();
            for (PreprocessingStrength rung : candidates) {
                boolean[] glareOut = new boolean[1];
                Mat map = EdgeMethodLab.strengthEdges(src, forceGlare, glareOut, rung);
                maps.put(rung, map);
                // Identical for every rung — the decision is made on the same gray with the
                // same pin — so the last one read is the one that ran.
                glare = glareOut[0];
                if (sizePrior == null) continue;

                // The detector's own geometry stage, so the boxes scored are the boxes the
                // inspection would get. Cloned because it is entitled to consume its input.
                Mat forBoxes = map.clone();
                List<RectF> rects = new ArrayList<>();
                try {
                    for (Rect r : HardwareFeatureExtractor.findFeatureBoxes(forBoxes, w, h)) {
                        rects.add(new RectF(r.x / w, r.y / h, (r.x + r.width) / w,
                                (r.y + r.height) / h));
                    }
                } finally {
                    forBoxes.release();
                }
                scores.put(rung, sizePrior.agreement(rects));
            }

            PreprocessingStrength chosen = sizePrior == null ? suggested
                    : choose(suggested, scores, AGREEMENT_MARGIN);
            Choice choice = new Choice(chosen, quality, suggested, scores, sizePrior);
            if (choiceOut != null && choiceOut.length > 0) choiceOut[0] = choice;
            Log.d(TAG, choice.describe());

            Mat map = maps.remove(chosen);
            HardwareFeatureExtractor.Edges edges = new HardwareFeatureExtractor.Edges(
                    gray, map, map.clone(), map.clone(), glare);
            ok = true;
            return edges;
        } finally {
            for (Mat m : maps.values()) m.release();
            if (!ok) gray.release();
        }
    }
}
