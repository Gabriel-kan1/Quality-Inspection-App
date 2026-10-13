package com.example.ml;

import android.graphics.RectF;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * How well a set of detections reproduces the operator's labelled reference, by where the
 * features are — the evidence {@link AdaptivePreprocessing} chooses an edge method on.
 *
 * <h3>Position first</h3>
 * Every labelled critical feature in the reference has a place on the part. A method that is
 * right for this photo puts a box there; one that is wrong either leaves the place empty (it
 * lost the feature — to a shadow, a merge, a faint edge) or puts boxes where the reference
 * has nothing (it invented hardware). Both are position errors, and position is what the
 * operator's labelling pins down most firmly, so it is the main score:
 *
 * <ul>
 *   <li>each reference feature costs its distance to the box matched to it, in units of
 *       {@link #REACH} times the feature's own size, from 0 (dead on) to 1;
 *   <li>a reference feature with no box within that reach costs 1: the method lost it;
 *   <li>an unmatched box whose centre lies on the part costs 1: the method invented it.
 * </ul>
 *
 * The score is the mean over all of those, so 0 is a box on every feature and nothing else,
 * and 1 is nothing right. Boxes are matched one to one, nearest first.
 *
 * <h3>Framing</h3>
 * The inspection is framed against the reference by the alignment gate, so the two photos
 * share a frame up to a small shift. The shift is estimated as the median offset from each
 * reference feature to its nearest box, and the better of "no shift" and that shift is kept.
 * There is no rotation or scale fit, deliberately: that needs four or more features to be
 * evidence (see {@link LayoutMatcher#tooSmallToRegister}), and a part with three — the
 * 3-hole plate — must still be scored. The size-based prior this replaced refused those parts
 * outright, so their inspections never ran the method chosen at reference setup.
 *
 * <h3>Then labels</h3>
 * Position cannot tell a box on the right place from a box on the right place around the
 * wrong thing. When a {@link Labeller} is attached, {@link #labelsAgreeing} classifies each
 * matched box and counts those whose label is the one the operator gave that feature; the
 * adaptive stage uses it to choose between methods that fit about equally well by position.
 */
public final class ReferenceFit {

    /**
     * How far a box's centre may sit from a reference feature and still be that feature, as a
     * multiple of the feature's larger side. One side: further than that and it is beside the
     * fastener, not on it.
     */
    public static final double REACH = 1.0;
    /** Size assumed for a feature whose reference box was not recorded (old layouts). */
    public static final float DEFAULT_FEATURE_SIDE = 0.05f;
    /** Smallest reach, so a tiny reference box does not demand pixel-perfect placement. */
    public static final float MIN_REACH = 0.02f;
    /** Largest framing shift looked for, as a fraction of the frame. */
    public static final double MAX_SHIFT = 0.15;
    /**
     * How far past the features' own bounding box a box still counts as "on the part", as a
     * fraction of that extent on each side — at least one feature's reach, so a part whose
     * features sit in a line still has a width.
     */
    public static final double ENVELOPE_PAD = 0.5;

    /** Classifies one box of the inspection photo; returns its type, or null when it cannot. */
    public interface Labeller {
        String typeOf(RectF box);
    }

    /** One fit of detections onto the reference. */
    public static final class Fit {
        /** 0 is every feature boxed and nothing else; 1 is nothing right. */
        public final double score;
        /** Index of the box matched to each reference feature, -1 where none was. */
        public final int[] boxForFeature;
        /** Reference features with a box. */
        public final int matched;
        /** Unmatched boxes on the part. */
        public final int extras;

        Fit(double score, int[] boxForFeature, int matched, int extras) {
            this.score = score;
            this.boxForFeature = boxForFeature;
            this.matched = matched;
            this.extras = extras;
        }
    }

    private final FeatureLayout layout;
    private final List<FeatureLayout.Slot> features;
    private final Labeller labeller;

    private ReferenceFit(FeatureLayout layout, Labeller labeller) {
        this.layout = layout;
        this.features = new ArrayList<>(layout.slots());
        this.labeller = labeller;
    }

    /** The fit for a labelled layout, or null when there is no layout or nothing in it. */
    public static ReferenceFit from(FeatureLayout layout) {
        if (layout == null || layout.isEmpty()) return null;
        return new ReferenceFit(layout, null);
    }

    /** The same reference, able to check labels with {@code labeller}. */
    public ReferenceFit withLabeller(Labeller labeller) {
        return new ReferenceFit(layout, labeller);
    }

    public boolean canCheckLabels() {
        return labeller != null;
    }

    /**
     * The method that drew the reference's boxes — what an inspection runs when nothing fits
     * better. Null when unknowable.
     */
    public String referenceMethodId() {
        return layout.referenceMethodId();
    }

    /** Labelled features scored against. */
    public int featureCount() {
        return features.size();
    }

    /** Mean position error of these detections against the reference, from 0 to 1. */
    public double agreement(List<RectF> rects) {
        return fit(rects).score;
    }

    /** Matches the detections onto the reference features by position and scores them. */
    public Fit fit(List<RectF> rects) {
        int n = features.size();
        if (rects == null || rects.isEmpty()) {
            int[] none = new int[n];
            Arrays.fill(none, -1);
            return new Fit(1.0, none, 0, 0);
        }
        float[] cx = new float[rects.size()];
        float[] cy = new float[rects.size()];
        for (int j = 0; j < rects.size(); j++) {
            cx[j] = (rects.get(j).left + rects.get(j).right) / 2f;
            cy[j] = (rects.get(j).top + rects.get(j).bottom) / 2f;
        }
        Fit still = fitAt(0f, 0f, cx, cy);
        float[] shift = estimateShift(cx, cy);
        if (shift == null) return still;
        Fit moved = fitAt(shift[0], shift[1], cx, cy);
        return moved.score < still.score ? moved : still;
    }

    /**
     * How many reference features have a matched box that classifies as the type the operator
     * gave that feature. -1 when no labeller is attached.
     */
    public int labelsAgreeing(List<RectF> rects, Fit fit) {
        if (labeller == null) return -1;
        int agree = 0;
        for (int i = 0; i < features.size(); i++) {
            int j = fit.boxForFeature[i];
            if (j < 0 || j >= rects.size()) continue;
            String type = labeller.typeOf(rects.get(j));
            if (type != null && type.equals(features.get(i).type)) agree++;
        }
        return agree;
    }

    public String describe() {
        return String.format(Locale.US, "%d labelled feature%s", features.size(),
                features.size() == 1 ? "" : "s");
    }

    private float reach(FeatureLayout.Slot s) {
        float side = s.hasSize() ? Math.max(s.w, s.h) : DEFAULT_FEATURE_SIDE;
        return Math.max(MIN_REACH, side * (float) REACH);
    }

    /** Median offset from each feature to its nearest box within {@link #MAX_SHIFT}. */
    private float[] estimateShift(float[] cx, float[] cy) {
        List<Float> dx = new ArrayList<>();
        List<Float> dy = new ArrayList<>();
        for (FeatureLayout.Slot s : features) {
            double best = MAX_SHIFT;
            int at = -1;
            for (int j = 0; j < cx.length; j++) {
                double d = Math.hypot(cx[j] - s.x, cy[j] - s.y);
                if (d < best) {
                    best = d;
                    at = j;
                }
            }
            if (at >= 0) {
                dx.add(cx[at] - s.x);
                dy.add(cy[at] - s.y);
            }
        }
        if (dx.isEmpty()) return null;
        return new float[]{median(dx), median(dy)};
    }

    private Fit fitAt(float dx, float dy, float[] cx, float[] cy) {
        int n = features.size();
        // Every feature-box pair within reach, nearest first, assigned one to one.
        List<double[]> pairs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            FeatureLayout.Slot s = features.get(i);
            float r = reach(s);
            float x = s.x + dx, y = s.y + dy;
            for (int j = 0; j < cx.length; j++) {
                double d = Math.hypot(cx[j] - x, cy[j] - y) / r;
                if (d < 1.0) pairs.add(new double[]{d, i, j});
            }
        }
        pairs.sort((a, b) -> Double.compare(a[0], b[0]));
        int[] boxFor = new int[n];
        Arrays.fill(boxFor, -1);
        double[] cost = new double[n];
        Arrays.fill(cost, 1.0);
        boolean[] used = new boolean[cx.length];
        int matched = 0;
        for (double[] p : pairs) {
            int i = (int) p[1], j = (int) p[2];
            if (boxFor[i] >= 0 || used[j]) continue;
            boxFor[i] = j;
            used[j] = true;
            cost[i] = p[0];
            matched++;
        }

        // The part's extent here: the features' bounding box, shifted and padded.
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        float maxReach = 0f;
        for (FeatureLayout.Slot s : features) {
            minX = Math.min(minX, s.x + dx);
            maxX = Math.max(maxX, s.x + dx);
            minY = Math.min(minY, s.y + dy);
            maxY = Math.max(maxY, s.y + dy);
            maxReach = Math.max(maxReach, reach(s));
        }
        float padX = Math.max((maxX - minX) * (float) ENVELOPE_PAD, maxReach);
        float padY = Math.max((maxY - minY) * (float) ENVELOPE_PAD, maxReach);
        int extras = 0;
        for (int j = 0; j < cx.length; j++) {
            if (used[j]) continue;
            if (cx[j] >= minX - padX && cx[j] <= maxX + padX
                    && cy[j] >= minY - padY && cy[j] <= maxY + padY) {
                extras++;
            }
        }

        double sum = extras;
        for (double c : cost) sum += c;
        return new Fit(sum / (n + extras), boxFor, matched, extras);
    }

    private static float median(List<Float> v) {
        List<Float> s = new ArrayList<>(v);
        s.sort(Float::compare);
        int m = s.size() / 2;
        return s.size() % 2 == 1 ? s.get(m) : (s.get(m - 1) + s.get(m)) / 2f;
    }

    /**
     * A labeller that classifies each distinct box once, however many methods put a box in
     * exactly the same place — the hybrids often agree to the pixel.
     */
    public static Labeller caching(Labeller inner) {
        Map<String, String> seen = new HashMap<>();
        return box -> {
            String key = box.left + "," + box.top + "," + box.right + "," + box.bottom;
            if (seen.containsKey(key)) return seen.get(key);
            String type = inner.typeOf(box);
            seen.put(key, type);
            return type;
        };
    }
}
