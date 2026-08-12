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
 * correspondences, derives the transform they imply, and counts how many other features then
 * land near a slot. The transform with the most agreement wins.
 *
 * This is deliberately robust rather than exact: missing, extra and misplaced features are
 * precisely what the inspection is looking for, so the alignment must tolerate them instead
 * of being derailed by them. A least-squares fit over all points would be dragged by the
 * very defects being detected.
 *
 * <h3>Registration is positional, never conditional on type</h3>
 * Inlier counting and slot assignment compare position only. That is what lets a stud sitting
 * in a screw's hole be identified as <em>that slot, wrong hardware</em> rather than as the
 * screw being missing and the stud being unexpected — which is the finding the whole
 * positional check exists to produce. Type agreement is used to <em>prioritise</em> which
 * correspondences to try first, never to decide what matches what.
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
    /**
     * Inliers required before an alignment is believed at all.
     *
     * Two is not a threshold, it is a tautology: a similarity transform has four degrees of
     * freedom and two point pairs supply exactly four constraints, so <em>any</em> two
     * correspondences fit perfectly and score two inliers. Accepting that meant a photo of a
     * different part could be "aligned" on a coincidence and then used to declare specific
     * slots missing — confident, precise, and wrong. Three is the smallest count with any
     * redundancy, so the third point is the first one that constitutes evidence.
     */
    private static final int MIN_INLIERS = 3;
    /**
     * ...and at least this share of the expected positions must be accounted for. Redundancy
     * alone is not enough on a large layout: three of twenty slots agreeing is still mostly
     * coincidence. Registering the part means recognising most of it.
     *
     * The cost is that a part missing more than half its hardware gets no positional report.
     * That is the right trade — a defect that gross is caught by the count comparison anyway,
     * and a misleading localisation is worse than an honest "cannot tell".
     */
    private static final float MIN_INLIER_FRACTION = 0.5f;

    /**
     * Inliers needed to believe an alignment of a layout this size.
     *
     * Capped at {@code slots - 1} so the requirement is always satisfiable while still leaving
     * at least one position free to be reported missing. Without the cap the absolute floor
     * dominates a small layout and demands perfection: a 3-slot layout needed 3 of 3, so a
     * single absent fastener made the check disappear entirely — exactly the case it exists
     * to report. A 2-slot layout could never align at all.
     *
     * Below {@link #MIN_INLIERS} slots there is nothing to cap toward, and the tautology the
     * floor guards against is unavoidable, so those layouts are refused outright by
     * {@link #tooSmallToRegister}.
     */
    public static int requiredInliers(int slots) {
        int wanted = Math.max(MIN_INLIERS, (int) Math.ceil(slots * MIN_INLIER_FRACTION));
        return Math.min(wanted, slots - 1);
    }

    /**
     * Whether a layout is too small for any alignment to constitute evidence.
     *
     * A similarity transform has four degrees of freedom and two point pairs supply exactly
     * four constraints, so two correspondences always fit perfectly. Corroborating that needs
     * a third point to exist in the first place, and one more beyond it to have anywhere to
     * report as missing.
     */
    public static boolean tooSmallToRegister(int slots) {
        return slots < MIN_INLIERS + 1;
    }

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
        /**
         * The winning similarity transform as {@code [a, b, tx, ty]}, or null when not aligned.
         * Retained so callers can ask where a slot lands in <em>this</em> photo — without it a
         * missing fastener can only be described in reference coordinates, which cannot be
         * drawn on the image the operator is actually looking at.
         */
        private final double[] transform;
        /** Diagnostics, so a refusal can say what was wrong rather than only that it failed. */
        public final int slotCount;
        public final int inlierCount;
        public final int requiredInliers;

        Result(boolean aligned, double[] transform, int slotCount, int inlierCount, int requiredInliers) {
            this.aligned = aligned;
            this.transform = transform;
            this.slotCount = slotCount;
            this.inlierCount = inlierCount;
            this.requiredInliers = requiredInliers;
        }

        /**
         * Maps a layout position into this inspection photo's fractional coordinates, or
         * returns null when no alignment was found.
         */
        public float[] project(float x, float y) {
            if (!aligned || transform == null) return null;
            return apply(transform, new float[]{x, y});
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
            int slots = layout == null ? 0 : layout.slots().size();
            return new Result(false, null, slots, 0, requiredInliers(Math.max(slots, 2)));
        }

        List<FeatureLayout.Slot> slots = layout.slots();
        float[][] slotPts = new float[slots.size()][];
        for (int i = 0; i < slots.size(); i++) {
            slotPts[i] = new float[]{slots.get(i).x, slots.get(i).y};
        }
        float[][] featPts = new float[features.size()][];
        for (int i = 0; i < features.size(); i++) featPts[i] = centre(features.get(i));

        double spread = spreadOf(slotPts);
        if (!(spread > 1e-6)) {
            return new Result(false, null, slots.size(), 0, requiredInliers(slots.size()));
        }
        double tolerance = spread * INLIER_TOLERANCE;

        // Two candidate pools for seeding the transform.
        //
        // Same-type pairs first: type is independent evidence and pruning by it makes a
        // random draw far more likely to hit a valid correspondence.
        //
        // But type agreement cannot be a *requirement*, because wrong hardware in the right
        // hole is one of the defects this check exists to find. Restricting the seed pool to
        // same-type pairs made alignment progressively less likely as substitutions grew —
        // the search starved exactly as the defect got worse — and a part with every fastener
        // swapped produced fewer than two candidates and no report at all. The second pool
        // drops the type constraint so position alone can carry the registration.
        List<int[]> typedCandidates = new ArrayList<>();
        List<int[]> allCandidates = new ArrayList<>();
        for (int s = 0; s < slots.size(); s++) {
            for (int f = 0; f < features.size(); f++) {
                allCandidates.add(new int[]{s, f});
                if (typesAgree(slots.get(s).type, types.get(f))) typedCandidates.add(new int[]{s, f});
            }
        }
        if (allCandidates.size() < 2) {
            return new Result(false, null, slots.size(), 0, requiredInliers(slots.size()));
        }

        Random rng = new Random(12345);
        double[] bestTransform = null;
        int bestInliers = 0;

        // Typed pass runs first and ties are kept (strict >), so an alignment that also agrees
        // on type wins over an equally-scoring one that only agrees on position.
        for (List<int[]> pool : java.util.Arrays.asList(typedCandidates, allCandidates)) {
            if (pool.size() < 2) continue;
            for (int iter = 0; iter < ITERATIONS; iter++) {
                int[] a = pool.get(rng.nextInt(pool.size()));
                int[] b = pool.get(rng.nextInt(pool.size()));
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
        }

        int required = requiredInliers(slots.size());
        if (bestTransform == null || bestInliers < required) {
            Log.d(TAG, "Alignment failed (best inliers " + bestInliers
                    + ", needed " + required + " of " + slots.size() + " slots)");
            return new Result(false, null, slots.size(), bestInliers, required);
        }

        // Assign greedily by ascending residual so the closest pairings claim their partners
        // first, and neither a slot nor a feature can be used twice.
        Result result = new Result(true, bestTransform, slots.size(), bestInliers, required);
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
