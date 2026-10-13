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
 * the operator labels every detection, and the critical ones become features with a position,
 * a box and a type. So when a labelled layout exists, <b>every method on the {@link #ballot}
 * runs on the photograph</b> and is judged in two steps:
 *
 * <ol>
 *   <li><b>Position.</b> {@link ReferenceFit} matches each method's boxes onto the reference
 *       features by where they are: a feature with no box near it, and a box on the part where
 *       the reference has nothing, both count against the method. This decides, and it works
 *       for a part of any size — three features as well as thirty.
 *   <li><b>Labels.</b> The methods within {@link #SHORTLIST_MARGIN} of the best position fit
 *       (at most {@link #MAX_LABELLED} of them) have their matched boxes classified, and the
 *       one whose boxes classify as the types the operator gave those features wins. Position
 *       says a box is in the right place; the label says it is around the right thing.
 * </ol>
 *
 * <p>Ties after both steps go to the method that drew the reference's own boxes
 * ({@link ReferenceFit#referenceMethodId}), so the reference and the inspection box the part
 * the same way unless another method is actually better here.
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
     * How close to the best position score a method must be to stay in the running for the
     * label check. On the lighting test photos every method that boxed all of a plate's
     * features scored within about 0.1 of the others, from where it centred its boxes: at
     * 0.05 the methods that labelled the 3-hole plate correctly under warm light from below
     * were cut before the labels were looked at, and at 0.10 they were kept (lighting variants
     * passed: 8, 9 and 6 of 9 against 7, 9 and 6). {@code python/variation_test_study.py}.
     */
    public static final double SHORTLIST_MARGIN = 0.10;
    /**
     * Most methods whose boxes are classified for the label check. Each classification is a
     * crop through the encoder; identical boxes from different methods are classified once.
     */
    public static final int MAX_LABELLED = 6;

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
        /** Position score per method tried, NaN where it failed; empty when no reference. */
        public final Map<String, Double> agreement;
        /**
         * Reference features each shortlisted method's boxes labelled correctly; empty when
         * labels were not checked.
         */
        public final Map<String, Integer> labels;
        /** The reference the methods were scored against, or null. */
        public final ReferenceFit fit;
        /** True when the operator pinned the method, so nothing was compared. */
        public final boolean pinned;

        Choice(String methodId, String incumbentId, ImageQuality quality,
               Map<String, Double> agreement, ReferenceFit fit, boolean pinned) {
            this(methodId, incumbentId, quality, agreement, new LinkedHashMap<>(), fit, pinned);
        }

        Choice(String methodId, String incumbentId, ImageQuality quality,
               Map<String, Double> agreement, Map<String, Integer> labels, ReferenceFit fit,
               boolean pinned) {
            this.methodId = methodId;
            this.incumbentId = incumbentId;
            this.quality = quality;
            this.agreement = Collections.unmodifiableMap(new LinkedHashMap<>(agreement));
            this.labels = Collections.unmodifiableMap(new LinkedHashMap<>(labels));
            this.fit = fit;
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
            sb.append(" — best position fit of ").append(agreement.size())
                    .append(" methods against the reference's ").append(fit.describe());
            Double own = agreement.get(methodId);
            if (own != null && !Double.isNaN(own)) {
                sb.append(String.format(Locale.US, " (score %.2f)", own));
            }
            Integer agree = labels.get(methodId);
            if (agree != null && agree >= 0) {
                sb.append("; of the ").append(labels.size()).append(" closest, its boxes matched ")
                        .append(agree).append(" of ").append(fit.featureCount())
                        .append(" labels");
            }
            if (overridden()) {
                Double inc = agreement.get(incumbentId);
                sb.append("; the reference's method, ").append(EdgeMethodLab.nameOf(incumbentId))
                        .append(inc == null || Double.isNaN(inc) ? ", did not run"
                                : String.format(Locale.US, ", scored %.2f", inc));
            } else {
                sb.append("; the reference's own method");
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
     * The boxes every competing method finds in this image, in catalogue order, as fractions.
     *
     * <p>For reference setup, which has nothing to score against: the operator flips between
     * these on the photo and their eye is the judge. Edge stage and geometry filter only — the
     * boxes an analysis would cut crops from, before anything classifies them.
     *
     * @param src working-resolution RGBA; not modified
     */
    public static Map<String, List<RectF>> survey(Mat src, Boolean forceGlare) {
        Map<String, List<RectF>> out = new LinkedHashMap<>();
        for (String id : ballot()) {
            List<RectF> rects = boxesFor(id, src, forceGlare, null);
            if (rects != null) out.put(id, rects);
        }
        return out;
    }

    /** How many boxes each method in a {@link #survey} found, in the same order. */
    public static Map<String, Integer> counts(Map<String, List<RectF>> boxes) {
        Map<String, Integer> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<RectF>> e : boxes.entrySet()) {
            out.put(e.getKey(), e.getValue().size());
        }
        return out;
    }

    /**
     * The order reference setup steps through the methods in: nearest the operator's count
     * first, so the plausible ones come before the ones that buried the part in boxes.
     * {@code counts}' own order among equals and throughout when there is no count, so the
     * first is always {@link #closestToCount}'s pick.
     */
    public static List<String> byCloseness(Map<String, Integer> counts, int target) {
        List<String> ids = new ArrayList<>(counts.keySet());
        if (target > 0) {
            // List.sort is stable, so equals keep the order they came in.
            ids.sort((a, b) -> Integer.compare(Math.abs(counts.get(a) - target),
                    Math.abs(counts.get(b) - target)));
        }
        return ids;
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
     * The methods close enough to the best position score to be decided between by labels:
     * within {@code margin} of it, best first, at most {@code max}. A method that failed (NaN
     * or null) is never on it.
     */
    static <K> List<K> shortlist(Map<K, Double> scores, double margin, int max) {
        List<Map.Entry<K, Double>> ranked = new ArrayList<>();
        for (Map.Entry<K, Double> e : scores.entrySet()) {
            if (e.getValue() != null && !Double.isNaN(e.getValue())) ranked.add(e);
        }
        ranked.sort((a, b) -> Double.compare(a.getValue(), b.getValue()));
        List<K> out = new ArrayList<>();
        if (ranked.isEmpty()) return out;
        double best = ranked.get(0).getValue();
        for (Map.Entry<K, Double> e : ranked) {
            if (out.size() >= max || e.getValue() > best + margin) break;
            out.add(e.getKey());
        }
        return out;
    }

    /**
     * The decision, separated from the OpenCV work so it can be tested.
     *
     * <ol>
     *   <li>Position: only the {@link #shortlist} — the methods within {@code margin} of the
     *       best position score — are in the running.
     *   <li>Labels: among those, the most reference features labelled correctly wins. Without
     *       label counts (none given, or a method missing from them) this step is skipped.
     *   <li>Ties: the incumbent if it is among the tied, otherwise the best position score.
     * </ol>
     *
     * Nothing scored: the incumbent stands.
     */
    static <K> K choose(K incumbent, Map<K, Double> scores, Map<K, Integer> labels,
                        double margin) {
        List<K> running = shortlist(scores, margin, Integer.MAX_VALUE);
        if (running.isEmpty()) return incumbent;
        if (labels != null && !labels.isEmpty()) {
            int most = Integer.MIN_VALUE;
            for (K k : running) {
                Integer v = labels.get(k);
                if (v != null) most = Math.max(most, v);
            }
            if (most != Integer.MIN_VALUE) {
                List<K> tied = new ArrayList<>();
                for (K k : running) {
                    Integer v = labels.get(k);
                    if (v != null && v == most) tied.add(k);
                }
                running = tied;
            }
        }
        return running.contains(incumbent) ? incumbent : running.get(0);
    }

    /**
     * Runs the adaptive edge stage.
     *
     * @param src        working-resolution RGBA image; not modified
     * @param forceGlare pins the black-hat glare branch, as for every other method
     * @param fit        the operator's labelled part to score against, or null; with a
     *                   labeller attached, labels decide between close position fits
     * @param choiceOut  one-element array that receives what was chosen and why; may be null
     * @param pinnedId   a catalogue id the operator chose, run without comparison; or null
     * @return the maps, owned by the caller
     * @throws IllegalStateException when the method that has to run produced nothing
     */
    static HardwareFeatureExtractor.Edges compute(Mat src, Boolean forceGlare,
                                                  ReferenceFit fit, Choice[] choiceOut,
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
        if (fit != null && fit.referenceMethodId() != null
                && ballot.contains(fit.referenceMethodId())) {
            incumbent = fit.referenceMethodId();
        }

        long startedAt = System.currentTimeMillis();
        if (pinnedId != null || fit == null) {
            String only = pinnedId != null ? pinnedId : incumbent;
            HardwareFeatureExtractor.Edges e =
                    EdgeMethodLab.computeWith(EdgeMethodLab.byId(only), src, forceGlare);
            Choice choice = new Choice(only, only, quality, new LinkedHashMap<>(), null,
                    pinnedId != null);
            if (choiceOut != null && choiceOut.length > 0) choiceOut[0] = choice;
            if (e == null) {
                throw new IllegalStateException(EdgeMethodLab.nameOf(only) + " produced no map");
            }
            return e;
        }

        // Only the boxes are kept per method: three working-resolution maps for each of
        // eighteen methods would be most of a hundred megabytes. The winner runs again.
        Map<String, List<RectF>> boxes = new LinkedHashMap<>();
        Map<String, ReferenceFit.Fit> fits = new LinkedHashMap<>();
        Map<String, Double> scores = new LinkedHashMap<>();
        for (String id : ballot) {
            List<RectF> rects = boxesFor(id, src, forceGlare, null);
            if (rects == null) {
                scores.put(id, Double.NaN);
                continue;
            }
            ReferenceFit.Fit f = fit.fit(rects);
            boxes.put(id, rects);
            fits.put(id, f);
            scores.put(id, f.score);
        }

        Map<String, Integer> labels = new LinkedHashMap<>();
        if (fit.canCheckLabels()) {
            List<String> close = shortlist(scores, SHORTLIST_MARGIN, MAX_LABELLED);
            if (close.size() > 1) {
                for (String id : close) {
                    labels.put(id, fit.labelsAgreeing(boxes.get(id), fits.get(id)));
                }
            }
        }
        String chosen = choose(incumbent, scores, labels, SHORTLIST_MARGIN);
        Choice choice = new Choice(chosen, incumbent, quality, scores, labels, fit, false);
        if (choiceOut != null && choiceOut.length > 0) choiceOut[0] = choice;
        Log.d(TAG, choice.describe() + String.format(Locale.US, " [%d ms for %d methods]",
                System.currentTimeMillis() - startedAt, ballot.size()));

        HardwareFeatureExtractor.Edges out =
                EdgeMethodLab.computeWith(EdgeMethodLab.byId(chosen), src, forceGlare);
        if (out == null) {
            throw new IllegalStateException(EdgeMethodLab.nameOf(chosen) + " produced no map");
        }
        return out;
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
