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
 * The edge stage that chooses its edge method per photograph, by scoring what each method
 * actually produced against what the operator said the part looks like.
 *
 * <h3>The only ground truth is the operator's</h3>
 * Nothing in a photograph says which edge method is right for it. The earlier design tried to
 * infer it from statistics of the pixels — a noise estimate, an edge-contrast estimate — with
 * thresholds fitted offline against hand counts of the bundled photographs. That put a
 * hard-coded ground truth into the runtime by the back door, it only ever chose between three
 * strengths of one method, and on 23 of 26 photographs it chose the same one. It adapted to
 * nothing.
 *
 * <p>The ground truth a real deployment has is the reference layout: during reference setup
 * the operator labels every detection, and the critical ones become slots with a position and
 * a box. So when a labelled layout exists, <b>every method in the Edge Lab catalogue runs on
 * the photograph</b>, each one's boxes are aligned onto the layout by {@link LayoutMatcher}
 * and scored by {@link SizePrior} — a box the wrong size, a slot nothing matched and a box
 * invented on the part all cost on one scale — and the method that reproduces the operator's
 * part best runs. The shipped stack, the tuned hybrids, the bare operators and the three
 * strengths all compete on equal terms, on every image.
 *
 * <p>The incumbent is the method that drew the reference's own boxes
 * ({@link SizePrior#referenceMethodId}); another displaces it only by beating it by
 * {@link #AGREEMENT_MARGIN}. Size jitter between two genuine detections of the same hardware
 * is a median 0.058 in these units ({@code python/size_ratio_study.py}), so a smaller
 * advantage is not evidence.
 *
 * <h3>Without a layout</h3>
 * Reference setup is the step that creates the layout, so there it has nothing to score
 * against. {@link ImageQuality}'s suggestion runs as a starting point only, and the operator
 * — the one party who can see which boxes are hardware — is shown what every other method
 * finds and can re-run with any of them ({@link #survey}). Whatever they settle on is recorded
 * in the layout and becomes the incumbent for every inspection that follows.
 *
 * <p>Cost: every catalogued method runs its edge stage and the geometry filter, plus one
 * alignment each — measured on the emulator in the results log. The crops and the encoder,
 * which dominate an inspection, run once, for the winner only.
 */
public final class AdaptivePreprocessing {

    private static final String TAG = "AdaptivePreprocessing";

    /**
     * How much lower (in {@code |ln ratio|} units) another method's score must be before it
     * displaces the incumbent.
     */
    public static final double AGREEMENT_MARGIN = 0.08;

    private AdaptivePreprocessing() {}

    /** What ran and why, for the results banner and the log. */
    public static final class Choice {
        /** Catalogue id of the method that ran. */
        public final String methodId;
        /**
         * What would have run without evidence against it: the reference's own method when a
         * layout names one, otherwise the image-quality suggestion.
         */
        public final String incumbentId;
        public final ImageQuality quality;
        /** Score per method tried, NaN where it could not be aligned; empty when no prior. */
        public final Map<String, Double> agreement;
        /** The prior the methods were scored against, or null. */
        public final SizePrior sizePrior;
        /** True when the operator pinned the method, so nothing was compared. */
        public final boolean pinned;

        Choice(String methodId, String incumbentId, ImageQuality quality,
               Map<String, Double> agreement, SizePrior sizePrior, boolean pinned) {
            this.methodId = methodId;
            this.incumbentId = incumbentId;
            this.quality = quality;
            this.agreement = Collections.unmodifiableMap(new LinkedHashMap<>(agreement));
            this.sizePrior = sizePrior;
            this.pinned = pinned;
        }

        /** The strength that ran when it was one of the three rungs, else null. */
        public PreprocessingStrength rung() {
            return PreprocessingStrength.byId(methodId);
        }

        /**
         * What a layout records in its {@code preprocessing} field: the rung label for a rung
         * (as layouts always have), the catalogue id for any other method.
         */
        public String label() {
            PreprocessingStrength rung = rung();
            return rung != null ? rung.label : methodId;
        }

        /** Whether the scoring displaced the incumbent. */
        public boolean overridden() {
            return !methodId.equals(incumbentId);
        }

        /** One sentence naming the method and the evidence, for a banner. */
        public String describe() {
            StringBuilder sb = new StringBuilder(EdgeMethodLab.nameOf(methodId));
            if (pinned) {
                return sb.append(" — chosen by the operator").toString();
            }
            if (agreement.isEmpty()) {
                return sb.append(" — no labelled reference to score against, so the image-")
                        .append("quality starting point (").append(quality.describe())
                        .append(')').toString();
            }
            int scored = 0;
            for (Double v : agreement.values()) if (v != null && !Double.isNaN(v)) scored++;
            sb.append(" — best of ").append(agreement.size()).append(" methods against the ")
                    .append("reference (").append(sizePrior.describe()).append("; ")
                    .append(scored).append(" aligned)");
            Double own = agreement.get(methodId);
            if (own != null && !Double.isNaN(own)) {
                sb.append(String.format(Locale.US, ", score %.2f", own));
            }
            if (overridden()) {
                Double inc = agreement.get(incumbentId);
                sb.append("; the reference's method, ").append(EdgeMethodLab.nameOf(incumbentId))
                        .append(inc == null || Double.isNaN(inc) ? ", could not be aligned"
                                : String.format(Locale.US, ", scored %.2f", inc));
            } else {
                sb.append("; nothing beat it by ").append(AGREEMENT_MARGIN);
            }
            return sb.toString();
        }
    }

    /** Catalogue ids of every method that competes: the whole catalogue but this stage. */
    public static List<String> ballot() {
        List<String> ids = new ArrayList<>();
        for (EdgeMethodLab.Method m : EdgeMethodLab.catalogue()) {
            if (!EdgeMethodLab.ADAPTIVE_ID.equals(m.id)) ids.add(m.id);
        }
        return ids;
    }

    /**
     * How many detections every competing method finds in this image, in catalogue order.
     *
     * <p>For reference setup, which has nothing to score against: the counts, and the
     * operator's own eye on the crops, are all there is. Edge stage and geometry filter only.
     *
     * @param src working-resolution RGBA; not modified
     */
    public static Map<String, Integer> survey(Mat src, Boolean forceGlare) {
        Map<String, Integer> out = new LinkedHashMap<>();
        for (String id : ballot()) {
            List<RectF> rects = boxesFor(id, src, forceGlare, null);
            if (rects != null) out.put(id, rects.size());
        }
        return out;
    }

    /**
     * The method whose count is nearest the number the operator said the part has. Ties go to
     * the first in {@code counts}' order, so the caller's ordering is the tie-break. Null when
     * {@code counts} is empty or the target is not positive.
     */
    public static String closestToCount(Map<String, Integer> counts, int target) {
        if (counts == null || target <= 0) return null;
        String best = null;
        int bestGap = Integer.MAX_VALUE;
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            if (e.getValue() == null) continue;
            int gap = Math.abs(e.getValue() - target);
            if (gap < bestGap) {
                bestGap = gap;
                best = e.getKey();
            }
        }
        return best;
    }

    /**
     * The decision, separated from the OpenCV work so it can be tested: the incumbent stands
     * unless some other candidate's score is lower by more than {@code margin}. A candidate
     * that could not be aligned (NaN) is never chosen. If the incumbent itself could not be
     * aligned, or was not scored, the best aligned candidate wins outright — boxes that cannot
     * be registered onto the part found something other than the part.
     */
    static <K> K choose(K incumbent, Map<K, Double> scores, double margin) {
        K best = null;
        double bestScore = Double.POSITIVE_INFINITY;
        for (Map.Entry<K, Double> e : scores.entrySet()) {
            double v = e.getValue() == null ? Double.NaN : e.getValue();
            if (Double.isNaN(v)) continue;
            if (v < bestScore) {
                bestScore = v;
                best = e.getKey();
            }
        }
        if (best == null) return incumbent;
        Double own = scores.get(incumbent);
        if (own == null || Double.isNaN(own)) return best;
        return bestScore < own - margin ? best : incumbent;
    }

    /**
     * Runs the adaptive edge stage.
     *
     * @param src        working-resolution RGBA image; not modified
     * @param forceGlare pins the black-hat glare branch, as for every other method
     * @param sizePrior  the operator's labelled part to score against, or null
     * @param choiceOut  one-element array that receives what was chosen and why; may be null
     * @param pinnedId   a catalogue id the operator chose, run without comparison; or null
     * @return the maps, owned by the caller
     * @throws IllegalStateException when the method that has to run produced nothing
     */
    static HardwareFeatureExtractor.Edges compute(Mat src, Boolean forceGlare,
                                                  SizePrior sizePrior, Choice[] choiceOut,
                                                  String pinnedId) {
        ImageQuality quality;
        Mat gray = new Mat();
        try {
            Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);
            quality = ImageQuality.assess(gray);
        } finally {
            gray.release();
        }
        List<String> ballot = ballot();
        if (pinnedId != null && !ballot.contains(pinnedId)) pinnedId = null;

        String incumbent = quality.suggest().id;
        if (sizePrior != null && sizePrior.referenceMethodId() != null
                && ballot.contains(sizePrior.referenceMethodId())) {
            incumbent = sizePrior.referenceMethodId();
        }

        boolean compare = pinnedId == null && sizePrior != null;
        List<String> candidates = pinnedId != null ? Collections.singletonList(pinnedId)
                : compare ? ballot : Collections.singletonList(incumbent);

        Map<String, HardwareFeatureExtractor.Edges> edgesById = new LinkedHashMap<>();
        Map<String, Double> scores = new LinkedHashMap<>();
        long startedAt = System.currentTimeMillis();
        try {
            for (String id : candidates) {
                HardwareFeatureExtractor.Edges e =
                        EdgeMethodLab.computeWith(EdgeMethodLab.byId(id), src, forceGlare);
                if (e == null) {
                    if (compare) scores.put(id, Double.NaN);
                    continue;
                }
                edgesById.put(id, e);
                if (!compare) continue;
                List<RectF> rects = boxesOf(e.edgesClosed, src.width(), src.height());
                scores.put(id, sizePrior.agreement(rects));
                // Only the winner's maps are needed afterwards, and each costs three
                // working-resolution Mats; keep the incumbent and the current best only.
                pruneLosers(edgesById, scores, incumbent);
            }

            String chosen = compare ? choose(incumbent, scores, AGREEMENT_MARGIN)
                    : candidates.get(0);
            Choice choice = new Choice(chosen, pinnedId != null ? pinnedId : incumbent,
                    quality, scores, compare ? sizePrior : null, pinnedId != null);
            if (choiceOut != null && choiceOut.length > 0) choiceOut[0] = choice;
            Log.d(TAG, choice.describe() + String.format(Locale.US, " [%d ms for %d method%s]",
                    System.currentTimeMillis() - startedAt, candidates.size(),
                    candidates.size() == 1 ? "" : "s"));

            HardwareFeatureExtractor.Edges out = edgesById.remove(chosen);
            if (out == null) {
                throw new IllegalStateException(EdgeMethodLab.nameOf(chosen) + " produced no map");
            }
            return out;
        } finally {
            for (HardwareFeatureExtractor.Edges e : edgesById.values()) e.release();
        }
    }

    /** Frees every held map that can no longer win: all but the incumbent and the best. */
    private static void pruneLosers(Map<String, HardwareFeatureExtractor.Edges> held,
                                    Map<String, Double> scores, String incumbent) {
        String best = null;
        double bestScore = Double.POSITIVE_INFINITY;
        for (Map.Entry<String, Double> e : scores.entrySet()) {
            Double v = e.getValue();
            if (v != null && !Double.isNaN(v) && v < bestScore) {
                bestScore = v;
                best = e.getKey();
            }
        }
        List<String> drop = new ArrayList<>();
        for (String id : held.keySet()) {
            if (!id.equals(incumbent) && !id.equals(best)) drop.add(id);
        }
        for (String id : drop) held.remove(id).release();
    }

    /**
     * The boxes one method would hand the inspection, in fractional coordinates, or null when
     * the method failed. Also yields the edge map's ownership to nobody: it is released here.
     */
    static List<RectF> boxesFor(String id, Mat src, Boolean forceGlare, boolean[] glareOut) {
        HardwareFeatureExtractor.Edges e =
                EdgeMethodLab.computeWith(EdgeMethodLab.byId(id), src, forceGlare);
        if (e == null) return null;
        try {
            if (glareOut != null && glareOut.length > 0) glareOut[0] = e.glareCorrectionApplied;
            return boxesOf(e.edgesClosed, src.width(), src.height());
        } finally {
            e.release();
        }
    }

    /**
     * The detector's own geometry stage over a map, as fractions — the boxes an inspection
     * would get. The map is cloned because the geometry stage is entitled to consume it.
     */
    static List<RectF> boxesOf(Mat closed, float w, float h) {
        Mat forBoxes = closed.clone();
        List<RectF> rects = new ArrayList<>();
        try {
            for (Rect r : HardwareFeatureExtractor.findFeatureBoxes(forBoxes, w, h)) {
                rects.add(new RectF(r.x / w, r.y / h, (r.x + r.width) / w,
                        (r.y + r.height) / h));
            }
        } finally {
            forBoxes.release();
        }
        return rects;
    }
}
