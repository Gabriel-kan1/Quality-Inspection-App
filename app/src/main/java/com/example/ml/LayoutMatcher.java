package com.example.ml;

import android.graphics.RectF;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Establishes which detected feature corresponds to which slot of a {@link FeatureLayout}.
 *
 * <h3>The problem</h3>
 * The part is photographed freehand, so between two shots it shifts, rotates and changes
 * apparent size. Positions therefore cannot be compared directly — a feature at (0.31,
 * 0.44) in one photo and (0.52, 0.29) in the next may well be the same physical fastener.
 * Something has to recover the transform between the two views before any positional claim
 * can be made.
 *
 * <h3>Approach</h3>
 * RANSAC over a similarity transform (uniform scale, rotation, translation). Two matched
 * point pairs determine such a transform exactly, so the search samples two candidate
 * correspondences of matching type, derives the transform they imply, and counts how many
 * other features then land near a slot. The transform with the most agreement wins.
 *
 * This is deliberately robust rather than exact: missing, extra and misplaced features are
 * precisely what the inspection is looking for, so the alignment must tolerate them instead
 * of being derailed by them. A least-squares fit over all points would be dragged by the
 * very defects being detected.
 *
 * <h3>Why not use the calibration marker</h3>
 * Aligning by marker would be simpler, but would require it to sit in a fixed position
 * relative to the part — a jig or printed fixture. Registering against the features
 * themselves imposes nothing on how the operator works, and keeps positional checking
 * available in photos where no marker is present at all.
 */
public final class LayoutMatcher {

    private static final String TAG = "LayoutMatcher";

    /**
     * How near a feature must land to a slot to count as the same thing, as a fraction of
     * the layout's overall spread. Generous enough to absorb detection jitter and mild
     * perspective, tight enough that a fastener in the wrong hole does not match.
     */
    public static float INLIER_TOLERANCE = 0.12f;
    /** RANSAC iterations. The candidate space is small, so this is ample and still trivial. */
    private static final int ITERATIONS = 400;
    /** Below this many features on either side there is nothing meaningful to align. */
    private static final int MIN_FEATURES = 2;

    /** One detected feature paired with the slot it was identified as. */
    public static final class Match {
        public final int slotId;
        public final int featureIndex;
        /** Distance between the transformed feature and its slot, in layout units. */
        public final double residual;
        /** True when the feature's classified type differs from the slot's expected type. */
        public final boolean typeMismatch;

        Match(int slotId, int featureIndex, double residual, boolean typeMismatch) {
            this.slotId = slotId;
            this.featureIndex = featureIndex;
            this.residual = residual;
            this.typeMismatch = typeMismatch;
        }
    }

    public static final class Result {
        public final List<Match> matches = new ArrayList<>();
        /** Slots with nothing found at them — features that should be present and are not. */
        public final List<Integer> missingSlotIds = new ArrayList<>();
        /** Detections that correspond to no slot — unexpected hardware, or false positives. */
        public final List<Integer> unmatchedFeatureIndices = new ArrayList<>();
        /** Whether a transform was found at all. False means no positional claim is possible. */
        public final boolean aligned;

        Result(boolean aligned) {
            this.aligned = aligned;
        }

        public String describe() {
            if (!aligned) return "could not align to reference layout";
            return String.format(Locale.US, "%d matched, %d missing, %d unexpected",
                    matches.size(), missingSlotIds.size(), unmatchedFeatureIndices.size());
        }
    }

    private LayoutMatcher() {}

    /** Centre point of a detection, in the same fractional space the layout uses. */
    private static float[] centre(RectF r) {
        return new float[]{(r.left + r.right) / 2f, (r.top + r.bottom) / 2f};
    }

    /**
     * Aligns detections onto the layout and reports the correspondence.
     *
     * @param layout   the expected arrangement.
     * @param features detected features, in the inspection photo's fractional coordinates.
     * @param types    classified type per feature, index-aligned with {@code features}.
     */
    public static Result match(FeatureLayout layout, List<RectF> features, List<String> types) {
        if (layout == null || layout.isEmpty() || features == null
                || features.size() < MIN_FEATURES || layout.slots().size() < MIN_FEATURES) {
            return new Result(false);
        }

        List<FeatureLayout.Slot> slots = layout.slots();
        float[][] slotPts = new float[slots.size()][];
        for (int i = 0; i < slots.size(); i++) {
            slotPts[i] = new float[]{slots.get(i).x, slots.get(i).y};
        }
        float[][] featPts = new float[features.size()][];
        for (int i = 0; i < features.size(); i++) featPts[i] = centre(features.get(i));

        double spread = spreadOf(slotPts);
        if (!(spread > 1e-6)) return new Result(false);
        double tolerance = spread * INLIER_TOLERANCE;

        // Candidate correspondences, restricted to same-type pairs: type is independent
        // evidence and pruning by it makes the sampling far more likely to hit a valid pair.
        List<int[]> candidates = new ArrayList<>();
        for (int s = 0; s < slots.size(); s++) {
            for (int f = 0; f < features.size(); f++) {
                if (typesAgree(slots.get(s).type, types.get(f))) candidates.add(new int[]{s, f});
            }
        }
        if (candidates.size() < 2) return new Result(false);

        Random rng = new Random(12345);
        double[] bestTransform = null;
        int bestInliers = 0;

        for (int iter = 0; iter < ITERATIONS; iter++) {
            int[] a = candidates.get(rng.nextInt(candidates.size()));
            int[] b = candidates.get(rng.nextInt(candidates.size()));
            if (a[0] == b[0] || a[1] == b[1]) continue;

            double[] t = similarityFrom(slotPts[a[0]], slotPts[b[0]], featPts[a[1]], featPts[b[1]]);
            if (t == null) continue;

            int inliers = 0;
            for (int s = 0; s < slots.size(); s++) {
                float[] projected = apply(t, slotPts[s]);
                for (int f = 0; f < features.size(); f++) {
                    if (dist(projected, featPts[f]) <= tolerance) { inliers++; break; }
                }
            }
            if (inliers > bestInliers) {
                bestInliers = inliers;
                bestTransform = t;
            }
        }

        if (bestTransform == null || bestInliers < MIN_FEATURES) {
            Log.d(TAG, "Alignment failed (best inliers " + bestInliers + ")");
            return new Result(false);
        }

        // Assign greedily by ascending residual so the closest pairings claim their partners
        // first, and neither a slot nor a feature can be used twice.
        Result result = new Result(true);
        boolean[] slotUsed = new boolean[slots.size()];
        boolean[] featUsed = new boolean[features.size()];

        List<double[]> pairs = new ArrayList<>();
        for (int s = 0; s < slots.size(); s++) {
            float[] projected = apply(bestTransform, slotPts[s]);
            for (int f = 0; f < features.size(); f++) {
                double d = dist(projected, featPts[f]);
                if (d <= tolerance) pairs.add(new double[]{d, s, f});
            }
        }
        java.util.Collections.sort(pairs, (p, q) -> Double.compare(p[0], q[0]));

        for (double[] p : pairs) {
            int s = (int) p[1], f = (int) p[2];
            if (slotUsed[s] || featUsed[f]) continue;
            slotUsed[s] = true;
            featUsed[f] = true;
            boolean mismatch = !typesAgree(slots.get(s).type, types.get(f));
            result.matches.add(new Match(slots.get(s).id, f, p[0], mismatch));
        }

        for (int s = 0; s < slots.size(); s++) {
            if (!slotUsed[s]) result.missingSlotIds.add(slots.get(s).id);
        }
        for (int f = 0; f < features.size(); f++) {
            if (!featUsed[f]) result.unmatchedFeatureIndices.add(f);
        }

        Log.d(TAG, "Layout match: " + result.describe());
        return result;
    }

    /**
     * Whether a detected type may occupy a slot. An unclassified detection is allowed to
     * match: position is independent evidence, and refusing would report a fastener the app
     * merely failed to recognise as missing hardware, which is the more misleading error.
     */
    private static boolean typesAgree(String slotType, String featureType) {
        if (slotType == null || featureType == null) return false;
        if (ReferenceLibrary.UNKNOWN.equals(featureType)) return true;
        return slotType.equals(featureType);
    }

    /** RMS distance of a point set from its centroid — the layout's characteristic size. */
    private static double spreadOf(float[][] pts) {
        double cx = 0, cy = 0;
        for (float[] p : pts) { cx += p[0]; cy += p[1]; }
        cx /= pts.length; cy /= pts.length;
        double sum = 0;
        for (float[] p : pts) sum += (p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy);
        return Math.sqrt(sum / pts.length);
    }

    /**
     * Similarity transform mapping s1→f1 and s2→f2 exactly, as {@code [a, b, tx, ty]} where
     * the mapping is {@code (x,y) -> (a·x − b·y + tx, b·x + a·y + ty)}. The pair (a, b)
     * encodes scale and rotation together, which is why two point correspondences suffice.
     */
    private static double[] similarityFrom(float[] s1, float[] s2, float[] f1, float[] f2) {
        double sdx = s2[0] - s1[0], sdy = s2[1] - s1[1];
        double denom = sdx * sdx + sdy * sdy;
        if (denom < 1e-12) return null;
        double fdx = f2[0] - f1[0], fdy = f2[1] - f1[1];
        double a = (fdx * sdx + fdy * sdy) / denom;
        double b = (fdy * sdx - fdx * sdy) / denom;
        double tx = f1[0] - (a * s1[0] - b * s1[1]);
        double ty = f1[1] - (b * s1[0] + a * s1[1]);
        return new double[]{a, b, tx, ty};
    }

    private static float[] apply(double[] t, float[] p) {
        return new float[]{
                (float) (t[0] * p[0] - t[1] * p[1] + t[2]),
                (float) (t[1] * p[0] + t[0] * p[1] + t[3])
        };
    }

    private static double dist(float[] a, float[] b) {
        return Math.hypot(a[0] - b[0], a[1] - b[1]);
    }
}
