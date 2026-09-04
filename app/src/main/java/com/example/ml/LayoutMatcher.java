package com.example.ml;

import android.graphics.RectF;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

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
 * Correspondence first, transform second.
 *
 * A candidate correspondence set is grown on the {@link DistanceNetwork} — the matrix of every
 * feature's separation from every other feature, which is a property of the part rather than of
 * the photograph. Two provisional pairings fix a scale; a third feature joins only if its
 * separation from <em>both</em> already matches what the layout says, and each further member is
 * checked against every member already accepted. A proper similarity transform (uniform scale,
 * rotation, translation — never reflection) is then fitted by least squares over the whole
 * agreed set, and the fit is verified by counting how many slots it actually accounts for.
 *
 * This is deliberately robust rather than exact: missing, extra and misplaced features are
 * precisely what the inspection is looking for, so the alignment must tolerate them instead
 * of being derailed by them. A least-squares fit over <em>all</em> points would be dragged by
 * the very defects being detected, which is why the fit is confined to the agreed set.
 *
 * <h3>Why the transform is fitted rather than taken from the seed</h3>
 * Two point pairs determine a similarity transform exactly, so it is tempting to use the seed's
 * transform directly — and that is what this did. But "exactly" means the transform inherits the
 * measurement error of those two detections and carries it into every prediction it makes:
 * residuals, the ghost markers drawn over the photo, and the displacement offsets reported to
 * the operator all came from whichever two boxes happened to seed the winner. Fitting over the
 * consensus set averages that error down instead of enshrining it.
 *
 * <h3>Registration is positional, never conditional on type</h3>
 * Growth, inlier counting and slot assignment compare position only. That is what lets a stud
 * sitting in a screw's hole be identified as <em>that slot, wrong hardware</em> rather than as
 * the screw being missing and the stud being unexpected — which is the finding the whole
 * positional check exists to produce. Type agreement is used to <em>order</em> which seeds are
 * tried, and matters only when the seed budget truncates the search.
 *
 * <h3>Why the network does not replace the transform</h3>
 * Separations are invariant to reflection, so a network-only match would accept a mirror image
 * of the part as a perfect fit. The similarity transform fitted afterwards admits no reflection,
 * and the inlier count is taken against <em>it</em>, so a mirrored consensus set fails
 * verification. Correspondence from the network, verification from the transform.
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
     *
     * Expressed in <em>layout</em> units and converted into photo units per candidate transform
     * — see {@link #toleranceFor}.
     */
    public static final float INLIER_TOLERANCE = 0.12f;
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
     * Widest range of apparent scales a seed may imply before it is discarded unexamined.
     *
     * A seed pairs two slots with two detections, which fixes how much larger the part appears
     * in this photo than in the reference. The capture workflow constrains that heavily — the
     * alignment gate makes the operator frame the part like the ghost overlay — so a seed
     * claiming the part shrank to a fifth or grew fivefold is not a hypothesis worth the cost
     * of testing. This is the cheapest thing the distance network does and it removes the large
     * majority of nonsense seeds for one division each.
     *
     * Kept far wider than the gate actually allows, because the penalty for a band that is too
     * tight is a silent refusal to register rather than a visible wrong answer.
     */
    private static final double SCALE_MIN = 0.2;
    private static final double SCALE_MAX = 5.0;
    /**
     * Shortest slot pair, as a fraction of the layout's spread, that may seed a transform.
     *
     * The seed pair is the baseline the whole registration is levered off: rotation and scale
     * are derived from it, so an error of a given size in a detection's centre produces an
     * angular error inversely proportional to how far apart the two slots are. Two adjacent
     * fasteners are a short lever and a poor one.
     */
    private static final double MIN_BASELINE_FRACTION = 0.25;
    /**
     * How much tighter the network's agreement test is than the positional one it feeds.
     *
     * Agreeing on separations is a <em>necessary</em> condition for two features being the same
     * thing, not a sufficient one: a detection displaced along the line to its neighbour changes
     * that separation by its full error, but one displaced across it barely changes it at all.
     * Run at the same threshold, the network is therefore systematically the more permissive of
     * the two tests — and it is the one that decides what the transform is fitted over.
     *
     * That is not a theoretical worry. A fastener fitted a fifth of the way out of position was
     * admitted to the consensus at the edge of a 12% threshold, the least-squares fit stretched
     * to meet it — cutting its residual almost in half and dragging every other position off in
     * the process — and the defect was then reported as correctly fitted. Two points each
     * displaced by half the threshold in opposite directions produce a separation error of the
     * whole threshold, so halving it is what makes the two tests comparable rather than leaving
     * a gap for exactly that case.
     *
     * The result is conservative growth and generous verification: the fit is computed only over
     * features that agree closely, while {@link #countInliers} still credits anything within the
     * full positional tolerance. Tightening this costs nothing in what gets counted as present —
     * only in what is allowed to define the alignment.
     */
    private static final double GROWTH_TOLERANCE_RATIO = 0.5;
    /**
     * Seeds actually grown, per priority pool. The pools are type-agreeing seeds and the rest.
     *
     * The search used to be 400 random draws from a pool of every slot paired with every
     * detection, which on a twenty-slot layout is thin coverage of eight hundred candidates and
     * made the answer a function of {@code Random(12345)}. Enumeration is exhaustive below this
     * cap and evenly strided above it, so the result no longer depends on a seed value and
     * covers the candidate space rather than sampling it.
     */
    private static final int MAX_SEEDS_PER_POOL = 256;
    /**
     * Ceiling on seeds *visited* during enumeration, before the scale and baseline filters.
     *
     * Bounds the pathological case — a layout with many slots and a photo full of unclassified
     * detections — at a few million divisions, which is milliseconds. Beyond it the enumeration
     * strides evenly through the candidate space instead of walking it.
     */
    private static final long ENUMERATION_CAP = 4_000_000L;

    /**
     * Smallest layout that gets the 6-DOF affine model rather than the 4-DOF similarity.
     *
     * <h3>Why affine at all</h3>
     * Measured, not assumed — see {@code python/localisation_tolerance_study.py}. Over 18
     * asset pairs whose correspondence is established independently (SIFT keypoints plus a
     * RANSAC homography, which knows nothing about this class), a correctly identified
     * fastener lands this far from where the model puts it, in multiples of the layout's
     * spread:
     *
     * <pre>
     *                p50     p90     p95     p99    pairs registering at 0.12
     *   similarity   0.031   0.101   0.141   0.218   16/18
     *   affine       0.022   0.075   0.087   0.134   18/18
     * </pre>
     *
     * The shipping tolerance of 0.12 sits <em>below</em> the similarity model's own p95, so
     * it was rejecting about one correct match in ten. The leftover is not detector noise but
     * model error: a rigid transform implies one ratio for every pairwise separation, yet the
     * measured spread of those ratios is median x1.28 and reaches x2.80 — the anisotropic
     * foreshortening of a camera that is not perpendicular to the part. Affine absorbs it and
     * cuts the p95 residual by 38%, which is what lets 0.12 stand as a measured value rather
     * than a widened one.
     *
     * <h3>Why not always</h3>
     * Three correspondences determine an affine transform exactly, so four is the first count
     * that constitutes evidence — the same argument {@link #MIN_INLIERS} makes for two and
     * three under a similarity. A layout of four slots has nowhere to spend that extra
     * requirement: demanding four of four is a check that structurally cannot fail, which
     * {@link #tooSmallToRegister} exists to refuse. Below this size the layout keeps the
     * poorer model because it is the only falsifiable one available.
     *
     * The cost is deliberate and worth stating plainly: a five-slot layout now needs four
     * positions located rather than three, so a part missing two of five reports no position
     * at all where it previously named them.
     */
    private static final int AFFINE_MIN_SLOTS = 5;
    /**
     * Inliers that determine each model exactly, and therefore one below the first count that
     * is evidence of anything.
     */
    private static final int AFFINE_MIN_INLIERS = 4;
    /**
     * How far from uniform an affine transform may stretch before it is refused.
     *
     * Affine has the freedom to squash the layout onto a line and collect spurious inliers
     * along it — the failure mode a similarity cannot have, and the price of the extra two
     * degrees of freedom. The measured foreshortening on real pairs is about x1.13 between
     * axes, so this is generous by more than a factor of two while still ruling out a
     * collapse.
     */
    private static final double MAX_ANISOTROPY = 2.5;

    /** Whether a layout of this size can afford the richer transform. */
    private static boolean usesAffine(int slots) {
        return slots >= AFFINE_MIN_SLOTS;
    }

    /** Inliers that merely reproduce the model's own degrees of freedom. */
    private static int modelFloor(int slots) {
        return usesAffine(slots) ? AFFINE_MIN_INLIERS : MIN_INLIERS;
    }

    /**
     * Inliers needed to believe an alignment of a layout this size.
     *
     * Capped at {@code slots - 1} so the requirement is always satisfiable while still leaving
     * at least one position free to be reported missing. Without the cap the absolute floor
     * dominates a small layout and demands perfection: a 3-slot layout needed 3 of 3, so a
     * single absent fastener made the check disappear entirely — exactly the case it exists
     * to report. A 2-slot layout could never align at all.
     *
     * The floor itself depends on which transform the layout is large enough to support, since
     * it is counting the redundancy beyond what determines that transform exactly — see
     * {@link #AFFINE_MIN_SLOTS}.
     *
     * Below {@link #MIN_INLIERS} slots there is nothing to cap toward, and the tautology the
     * floor guards against is unavoidable, so those layouts are refused outright by
     * {@link #tooSmallToRegister}.
     */
    public static int requiredInliers(int slots) {
        int wanted = Math.max(modelFloor(slots), (int) Math.ceil(slots * MIN_INLIER_FRACTION));
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
         * The winning transform as {@code [a11, a12, a21, a22, tx, ty]}, or null when not
         * aligned. Retained so callers can ask where a slot lands in <em>this</em> photo —
         * without it a missing fastener can only be described in reference coordinates, which
         * cannot be drawn on the image the operator is actually looking at.
         *
         * Always the six-element affine form. A similarity is the constrained case
         * {@code a11 == a22, a12 == -a21}, so one representation serves both models and
         * projection does not need to know which was fitted.
         */
        private final double[] transform;
        /** Diagnostics, so a refusal can say what was wrong rather than only that it failed. */
        public final int slotCount;
        public final int inlierCount;
        public final int requiredInliers;
        /**
         * Worst disagreement inside the correspondence set between an observed separation and
         * the one the layout expects, as a fraction of the part's own size. NaN when nothing
         * was aligned.
         *
         * A second, independent reading of fit quality. The inlier count says how many
         * positions were accounted for but not how well: a set can clear the count while
         * sitting near the edge of tolerance throughout, which is what a photo taken at a
         * steep angle looks like. This number separates the two, and it is comparable across
         * parts because it is scaled by the part.
         */
        public final double distortion;

        Result(boolean aligned, double[] transform, int slotCount, int inlierCount,
               int requiredInliers, double distortion) {
            this.aligned = aligned;
            this.transform = transform;
            this.slotCount = slotCount;
            this.inlierCount = inlierCount;
            this.requiredInliers = requiredInliers;
            this.distortion = distortion;
        }

        /**
         * Maps a layout position into this inspection photo's fractional coordinates, or
         * returns null when no alignment was found.
         */
        public float[] project(float x, float y) {
            if (!aligned || transform == null) return null;
            return apply(transform, new float[]{x, y});
        }

        /**
         * The reverse of {@link #project}: maps a point in this photo back into layout
         * coordinates, or null when no alignment was found.
         *
         * Lets a caller measure how far a detection sits from the slot it probably belongs to
         * in the layout's own frame, which is the only frame where "a fifth of the way across
         * the part" means anything. Measuring in photo coordinates would fold the operator's
         * distance from the part into the answer.
         */
        public float[] unproject(float x, float y) {
            if (!aligned || transform == null) return null;
            double det = transform[0] * transform[3] - transform[1] * transform[2];
            if (Math.abs(det) < 1e-12) return null;
            double dx = x - transform[4], dy = y - transform[5];
            return new float[]{
                    (float) ((transform[3] * dx - transform[1] * dy) / det),
                    (float) ((transform[0] * dy - transform[2] * dx) / det)};
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
     * Matching tolerance in <em>photo</em> units for a given transform.
     *
     * {@link #INLIER_TOLERANCE} is a fraction of the part, and the part's size in the photo is
     * its size in the layout multiplied by the transform's scale. Using the layout figure
     * directly — which this did — silently retunes the check by how far away the operator
     * stood. Shooting the part at half the reference's apparent size doubled the tolerance
     * relative to the part, which is the direction that matters: a fastener in the wrong hole
     * then lands inside tolerance and is reported as correctly fitted.
     */
    private static double toleranceFor(double[] t, double layoutSpread) {
        return layoutSpread * INLIER_TOLERANCE * scaleOf(t);
    }

    /**
     * The isotropic scale a transform implies — the geometric mean of its two singular values.
     * For a similarity this is exactly {@code hypot(a, b)}; for an affine it is the factor by
     * which areas change, which is the honest single number when the two axes differ.
     */
    private static double scaleOf(double[] t) {
        return Math.sqrt(Math.abs(t[0] * t[3] - t[1] * t[2]));
    }

    /**
     * The rotation a transform is closest to, from the polar decomposition of its linear part.
     * Reduces to {@code atan2(b, a)} for a similarity, so the least-rotation tiebreak means
     * the same thing under both models.
     */
    private static double rotationOf(double[] t) {
        return Math.abs(Math.atan2(t[2] - t[1], t[0] + t[3]));
    }

    /**
     * Ratio of an affine's larger stretch to its smaller — 1 for any similarity. Closed-form
     * singular values of a 2x2, which is cheaper and steadier than a general SVD at this size.
     */
    private static double anisotropyOf(double[] t) {
        double e = (t[0] + t[3]) / 2, f = (t[0] - t[3]) / 2;
        double g = (t[2] + t[1]) / 2, h = (t[2] - t[1]) / 2;
        double q = Math.hypot(e, h), r = Math.hypot(f, g);
        double larger = q + r, smaller = Math.abs(q - r);
        return smaller < 1e-12 ? Double.MAX_VALUE : larger / smaller;
    }

    private static String typeAt(List<String> types, int index) {
        return types == null || index < 0 || index >= types.size() ? null : types.get(index);
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
            return new Result(false, null, slots, 0, requiredInliers(Math.max(slots, 2)),
                    Double.NaN);
        }

        List<FeatureLayout.Slot> slots = layout.slots();
        float[][] slotPts = new float[slots.size()][];
        for (int i = 0; i < slots.size(); i++) {
            slotPts[i] = new float[]{slots.get(i).x, slots.get(i).y};
        }
        float[][] featPts = new float[features.size()][];
        for (int i = 0; i < features.size(); i++) featPts[i] = centre(features.get(i));

        DistanceNetwork slotNet = DistanceNetwork.over(slotPts);
        DistanceNetwork featNet = DistanceNetwork.over(featPts);
        double spread = slotNet.spread();
        if (!(spread > 1e-6)) {
            return new Result(false, null, slots.size(), 0, requiredInliers(slots.size()),
                    Double.NaN);
        }

        List<int[]> seeds = seedsFor(slotNet, featNet, slots, types);
        if (seeds.isEmpty()) {
            Log.d(TAG, "No seed correspondence implies a plausible scale — the detections do not"
                    + " have the separations this layout describes.");
            return new Result(false, null, slots.size(), 0, requiredInliers(slots.size()),
                    Double.NaN);
        }

        DistanceNetwork.Matcher grower = new DistanceNetwork.Matcher(slotNet, featNet);
        boolean affine = usesAffine(slots.size());
        boolean[] used = new boolean[features.size()];
        int[] assignment = new int[slots.size()];

        double[] bestTransform = null;
        int bestInliers = 0;
        double bestRotation = Double.MAX_VALUE;
        double bestDistortion = Double.NaN;
        // Largest agreeing set any seed produced, so a refusal can say which stage fell short:
        // a consensus that never grew past its seed means the detections do not have the
        // layout's shape at all, which is a different problem from one that grew and then
        // failed verification.
        int bestConsensus = 0;

        // Every seed is scored on positional support alone. Ties are broken by the smaller
        // rotation, never by type agreement.
        //
        // Preferring the type-consistent alignment on a tie looked harmless and is in fact the
        // one rule guaranteed to hide the defect being looked for. A near-symmetric layout —
        // four fasteners at the corners of a parallelogram — maps onto itself under a half
        // turn, so "every fastener swapped" and "nothing wrong, photographed upside down" have
        // identical positional support. Given the choice, the typed preference took the second
        // reading every time and reported a fully swapped part as all present.
        //
        // Least rotation is the honest tiebreak here because the capture workflow already
        // constrains it: the alignment gate makes the operator frame the part like the ghost
        // overlay, so a half-turn interpretation is a priori implausible in a way a small
        // freehand rotation is not.
        for (int[] seed : seeds) {
            DistanceNetwork.Consensus consensus = grower.grow(seed[0], seed[1], seed[2], seed[3],
                    spread * INLIER_TOLERANCE * GROWTH_TOLERANCE_RATIO
                            * impliedScale(slotNet, featNet, seed));
            if (consensus == null) continue;
            bestConsensus = Math.max(bestConsensus, consensus.size);

            double[] t = fitSimilarity(slotPts, featPts, consensus.observedForNode);
            if (t == null) continue;

            // Refine once against the transform's own inliers. Growth admits a member on its
            // separations alone, which cannot distinguish a point from its mirror across the
            // line through the members already accepted; the fitted transform can, so re-fitting
            // on what it actually agrees with drops any such member back out.
            //
            // The refinement is also where the layout's own model is applied. The seed and the
            // growth are always similarity-based — two correspondences cannot say anything
            // about shear — so affine enters only once there is a correspondence set worth
            // fitting it to, and only for layouts with the slots to make four inliers evidence
            // rather than arithmetic.
            int inliers = countInliers(t, slotPts, featPts, toleranceFor(t, spread), used,
                    assignment);
            double[] refined = affine ? fitAffine(slotPts, featPts, assignment) : null;
            if (refined == null) refined = fitSimilarity(slotPts, featPts, assignment);
            if (refined != null) {
                int refinedInliers = countInliers(refined, slotPts, featPts,
                        toleranceFor(refined, spread), used, null);
                if (refinedInliers >= inliers) {
                    t = refined;
                    inliers = refinedInliers;
                }
            }

            if (inliers < bestInliers) continue;
            double rotation = rotationOf(t);
            if (inliers > bestInliers || rotation < bestRotation) {
                bestInliers = inliers;
                bestRotation = rotation;
                bestTransform = t;
                bestDistortion = consensus.distortion / Math.max(consensus.scale * spread, 1e-9);
            }
        }

        int required = requiredInliers(slots.size());
        if (bestTransform == null || bestInliers < required) {
            Log.d(TAG, String.format(Locale.US,
                    "Alignment failed (best inliers %d, needed %d of %d slots; %d candidate"
                            + " detections, %d seeds tried, best consensus %d, spread %.4f)",
                    bestInliers, required, slots.size(), features.size(), seeds.size(),
                    bestConsensus, spread));
            // What was actually on offer. A refusal that lists only the slots leaves the two
            // causes indistinguishable from the log: a layout the detections cannot support,
            // versus detections the search failed to pair up.
            for (int s = 0; s < slots.size(); s++) {
                Log.d(TAG, String.format(Locale.US, "  slot %d (%s) wants (%.3f, %.3f)",
                        slots.get(s).id, slots.get(s).type, slotPts[s][0], slotPts[s][1]));
            }
            for (int f = 0; f < featPts.length; f++) {
                Log.d(TAG, String.format(Locale.US, "  candidate %d (%s) at (%.3f, %.3f)",
                        f, typeAt(types, f), featPts[f][0], featPts[f][1]));
            }
            logResiduals(bestTransform, slotPts, featPts);
            return new Result(false, null, slots.size(), bestInliers, required, Double.NaN);
        }

        double tolerance = toleranceFor(bestTransform, spread);

        // Assign greedily by ascending residual so the closest pairings claim their partners
        // first, and neither a slot nor a feature can be used twice.
        Result result = new Result(true, bestTransform, slots.size(), bestInliers, required,
                bestDistortion);
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
            boolean mismatch = !typesAgree(slots.get(s).type, typeAt(types, f));
            result.matches.add(new Match(slots.get(s).id, f, p[0], mismatch));
        }

        for (int s = 0; s < slots.size(); s++) {
            if (!slotUsed[s]) result.missingSlotIds.add(slots.get(s).id);
        }
        for (int f = 0; f < features.size(); f++) {
            if (!featUsed[f]) result.unmatchedFeatureIndices.add(f);
        }

        Log.d(TAG, String.format(Locale.US, "Layout match: %s (scale %.3f, rotation %.1f°,"
                        + " network distortion %.1f%% of the part)",
                result.describe(), scaleOf(bestTransform), Math.toDegrees(bestRotation),
                bestDistortion * 100));
        for (Match m : result.matches) {
            FeatureLayout.Slot slot = layout.slotById(m.slotId);
            Log.d(TAG, String.format(Locale.US,
                    "  slot %d (%s) <- detection %d (%s), residual %.4f%s",
                    m.slotId, slot == null ? "?" : slot.type, m.featureIndex,
                    typeAt(types, m.featureIndex), m.residual,
                    m.typeMismatch ? "   TYPE MISMATCH" : ""));
        }
        for (Integer slotId : result.missingSlotIds) {
            FeatureLayout.Slot slot = layout.slotById(slotId);
            Log.d(TAG, "  slot " + slotId + " (" + (slot == null ? "?" : slot.type)
                    + ") matched nothing");
        }
        for (Integer f : result.unmatchedFeatureIndices) {
            Log.d(TAG, String.format(Locale.US, "  detection %d (%s) at (%.3f, %.3f) unexpected",
                    f, typeAt(types, f), featPts[f][0], featPts[f][1]));
        }
        logResiduals(bestTransform, slotPts, featPts);
        return result;
    }

    private static double impliedScale(DistanceNetwork slotNet, DistanceNetwork featNet,
                                       int[] seed) {
        return featNet.between(seed[1], seed[3]) / slotNet.between(seed[0], seed[2]);
    }

    /**
     * Candidate seed correspondences, as {@code {slotA, featureA, slotB, featureB}}, ordered
     * with type-agreeing seeds first.
     *
     * <h3>Why enumerate rather than sample</h3>
     * This was 400 draws from {@code Random(12345)} over every slot-detection pairing. Two
     * problems: on anything but a small layout that is sparse coverage of the candidate space,
     * and the answer was a function of the seed constant — a different constant gave a
     * different registration on the same photo. The distance network makes enumeration
     * affordable by rejecting most candidates for the cost of one division: a seed whose two
     * separations imply the part changed apparent size fivefold cannot be a correspondence,
     * whatever else is true of it, and never needs a transform built for it.
     *
     * <h3>Type ordering</h3>
     * Type is independent evidence and a type-agreeing seed is far likelier to be a real
     * correspondence, so those are tried first. It cannot be a <em>requirement</em>: wrong
     * hardware in the right hole is one of the defects this check exists to find, and
     * restricting the seeds to same-type pairs made alignment starve exactly as substitutions
     * grew — a part with every fastener swapped produced no valid seed and no report at all.
     * The ordering only decides what gets dropped when the budget truncates.
     */
    private static List<int[]> seedsFor(DistanceNetwork slotNet, DistanceNetwork featNet,
                                        List<FeatureLayout.Slot> slots, List<String> types) {
        // Slot pairs far enough apart to be a usable baseline, computed once.
        List<int[]> baselines = new ArrayList<>();
        double minBaseline = MIN_BASELINE_FRACTION * slotNet.spread();
        for (int i = 0; i < slotNet.size(); i++) {
            for (int j = i + 1; j < slotNet.size(); j++) {
                if (slotNet.between(i, j) >= minBaseline) baselines.add(new int[]{i, j});
            }
        }
        if (baselines.isEmpty()) return new ArrayList<>();

        int f = featNet.size();
        long orderedFeaturePairs = (long) f * (f - 1);
        long total = (long) baselines.size() * orderedFeaturePairs;
        if (total <= 0) return new ArrayList<>();
        // Stride evenly rather than walking the space when it is large, so coverage stays
        // uniform across the whole candidate set instead of exhausting the first slot pairs.
        long step = Math.max(1L, total / ENUMERATION_CAP);

        List<int[]> typed = new ArrayList<>();
        List<int[]> untyped = new ArrayList<>();
        for (long n = 0; n < total; n += step) {
            int[] base = baselines.get((int) (n / orderedFeaturePairs));
            long fp = n % orderedFeaturePairs;
            int p = (int) (fp / (f - 1));
            int k = (int) (fp % (f - 1));
            int q = k < p ? k : k + 1;

            double scale = featNet.between(p, q) / slotNet.between(base[0], base[1]);
            if (!(scale >= SCALE_MIN) || !(scale <= SCALE_MAX)) continue;

            int[] seed = new int[]{base[0], p, base[1], q};
            if (typesAgree(slots.get(base[0]).type, typeAt(types, p))
                    && typesAgree(slots.get(base[1]).type, typeAt(types, q))) {
                typed.add(seed);
            } else {
                untyped.add(seed);
            }
        }

        List<int[]> seeds = decimate(typed, MAX_SEEDS_PER_POOL);
        seeds.addAll(decimate(untyped, MAX_SEEDS_PER_POOL));
        return seeds;
    }

    /** Evenly thins a candidate list down to {@code limit}, keeping coverage of the whole. */
    private static List<int[]> decimate(List<int[]> candidates, int limit) {
        if (candidates.size() <= limit) return new ArrayList<>(candidates);
        List<int[]> kept = new ArrayList<>(limit);
        // Rational stride rather than an integer one, so the samples stay spread across the
        // whole list instead of clustering at the front when the ratio is not a whole number.
        for (int i = 0; i < limit; i++) {
            kept.add(candidates.get((int) ((long) i * candidates.size() / limit)));
        }
        return kept;
    }

    /**
     * Least-squares proper similarity transform mapping the assigned slots onto their features.
     *
     * "Proper" is the load-bearing word: the solution is constrained to scale, rotation and
     * translation, with no reflection available to it. The {@link DistanceNetwork} growth that
     * produces {@code observedForNode} cannot tell a part from its mirror image, because
     * separations are identical either way. This fit can, and the inlier count taken against it
     * is what turns that into a refusal.
     *
     * @param observedForNode feature index per slot, -1 where unassigned.
     * @return {@code [a11, a12, a21, a22, tx, ty]}, or null when fewer than two slots are
     *         assigned or they are coincident.
     */
    private static double[] fitSimilarity(float[][] slotPts, float[][] featPts,
                                          int[] observedForNode) {
        double[] c = centroids(slotPts, featPts, observedForNode, 2);
        if (c == null) return null;
        double sx = c[0], sy = c[1], fx = c[2], fy = c[3];

        double numA = 0, numB = 0, den = 0;
        for (int k = 0; k < observedForNode.length; k++) {
            int r = observedForNode[k];
            if (r < 0) continue;
            double ux = slotPts[k][0] - sx, uy = slotPts[k][1] - sy;
            double vx = featPts[r][0] - fx, vy = featPts[r][1] - fy;
            numA += ux * vx + uy * vy;
            numB += ux * vy - uy * vx;
            den += ux * ux + uy * uy;
        }
        if (den < 1e-12) return null;
        double a = numA / den, b = numB / den;
        if (!(Math.hypot(a, b) > 1e-9)) return null;
        return new double[]{a, -b, b, a, fx - (a * sx - b * sy), fy - (b * sx + a * sy)};
    }

    /**
     * Least-squares affine transform over the assigned slots — six degrees of freedom, so the
     * two axes may scale differently and shear.
     *
     * That extra freedom is the point: the residual a similarity leaves on real photo pairs is
     * dominated by the anisotropic foreshortening of a camera that is not square to the part,
     * which no rigid transform can express at any tolerance. It is also the risk, since an
     * affine can squash the layout onto a line and harvest inliers along it — hence the
     * {@link #MAX_ANISOTROPY} guard, and hence requiring four assignments so the fit is never
     * merely reproducing its own degrees of freedom.
     *
     * @return {@code [a11, a12, a21, a22, tx, ty]}, or null when under-determined, degenerate
     *         or stretched beyond what a photograph of a rigid part can produce.
     */
    private static double[] fitAffine(float[][] slotPts, float[][] featPts,
                                      int[] observedForNode) {
        double[] c = centroids(slotPts, featPts, observedForNode, AFFINE_MIN_INLIERS);
        if (c == null) return null;
        double sx = c[0], sy = c[1], fx = c[2], fy = c[3];

        double sxx = 0, sxy = 0, syy = 0, bx = 0, by = 0, cx = 0, cy = 0;
        for (int k = 0; k < observedForNode.length; k++) {
            int r = observedForNode[k];
            if (r < 0) continue;
            double ux = slotPts[k][0] - sx, uy = slotPts[k][1] - sy;
            double vx = featPts[r][0] - fx, vy = featPts[r][1] - fy;
            sxx += ux * ux;
            sxy += ux * uy;
            syy += uy * uy;
            bx += ux * vx;
            by += uy * vx;
            cx += ux * vy;
            cy += uy * vy;
        }
        double d = sxx * syy - sxy * sxy;
        // Collinear slots leave the normal equations singular: a layout whose hardware sits in
        // a straight line cannot pin a transform across the line, and no amount of it will.
        if (Math.abs(d) < 1e-12) return null;

        double a11 = (bx * syy - by * sxy) / d, a12 = (sxx * by - sxy * bx) / d;
        double a21 = (cx * syy - cy * sxy) / d, a22 = (sxx * cy - sxy * cx) / d;
        double[] t = {a11, a12, a21, a22,
                fx - (a11 * sx + a12 * sy), fy - (a21 * sx + a22 * sy)};
        if (!(scaleOf(t) > 1e-9) || anisotropyOf(t) > MAX_ANISOTROPY) return null;
        return t;
    }

    /** Slot and feature centroids over the assignment, or null below {@code minimum} pairs. */
    private static double[] centroids(float[][] slotPts, float[][] featPts,
                                      int[] observedForNode, int minimum) {
        if (observedForNode == null) return null;
        double sx = 0, sy = 0, fx = 0, fy = 0;
        int n = 0;
        for (int k = 0; k < observedForNode.length; k++) {
            int r = observedForNode[k];
            if (r < 0) continue;
            sx += slotPts[k][0];
            sy += slotPts[k][1];
            fx += featPts[r][0];
            fy += featPts[r][1];
            n++;
        }
        if (n < minimum) return null;
        return new double[]{sx / n, sy / n, fx / n, fy / n};
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

    /**
     * Prints where the best transform put each slot and how far the nearest detection was.
     *
     * A bare inlier count says an alignment was refused but not why. These residuals separate
     * the two causes that look identical from the outside: a part that genuinely does not
     * match, versus a good correspondence sitting just outside a tolerance derived from a
     * layout whose spread is small.
     */
    private static void logResiduals(double[] t, float[][] slotPts, float[][] featPts) {
        if (t == null) {
            Log.d(TAG, "  no transform survived — not enough distinct correspondences to sample");
            return;
        }
        for (int s = 0; s < slotPts.length; s++) {
            float[] projected = apply(t, slotPts[s]);
            double best = Double.MAX_VALUE;
            for (float[] featPt : featPts) best = Math.min(best, dist(projected, featPt));
            Log.d(TAG, String.format(Locale.US,
                    "  slot %d (%.3f, %.3f) -> (%.3f, %.3f), nearest detection %.4f away",
                    s + 1, slotPts[s][0], slotPts[s][1], projected[0], projected[1], best));
        }
    }

    /**
     * How many slots a transform accounts for, counted <em>one-to-one</em>: each detection can
     * satisfy at most one slot.
     *
     * This has to use the same rule as the assignment below, because the count is what decides
     * whether the alignment is believed. Counting "does any detection land near this slot"
     * let a single dense cluster satisfy many slots at once, so a transform that collapsed the
     * layout onto a clump of scratches scored full marks and was accepted — and then the
     * one-to-one assignment could only pair a couple of them, reporting the rest as missing on
     * a part whose counts matched exactly.
     *
     * Greedy nearest-first per slot rather than a globally optimal matching: this runs inside
     * the seed loop, and the gate only needs a count that cannot exceed what assignment will
     * actually achieve.
     *
     * @param assignment optional out-parameter filled with the feature chosen per slot, -1
     *                   where none. Lets the caller re-fit the transform on exactly what it
     *                   agreed with.
     */
    private static int countInliers(double[] t, float[][] slotPts, float[][] featPts,
                                    double tolerance, boolean[] used, int[] assignment) {
        java.util.Arrays.fill(used, false);
        if (assignment != null) java.util.Arrays.fill(assignment, -1);
        int inliers = 0;
        for (int s = 0; s < slotPts.length; s++) {
            float[] projected = apply(t, slotPts[s]);
            int best = -1;
            double bestDist = tolerance;
            for (int f = 0; f < featPts.length; f++) {
                if (used[f]) continue;
                double d = dist(projected, featPts[f]);
                if (d <= bestDist) {
                    bestDist = d;
                    best = f;
                }
            }
            if (best >= 0) {
                used[best] = true;
                if (assignment != null) assignment[s] = best;
                inliers++;
            }
        }
        return inliers;
    }

    /** Maps a layout point through {@code [a11, a12, a21, a22, tx, ty]}. */
    private static float[] apply(double[] t, float[] p) {
        return new float[]{
                (float) (t[0] * p[0] + t[1] * p[1] + t[4]),
                (float) (t[2] * p[0] + t[3] * p[1] + t[5])
        };
    }

    private static double dist(float[] a, float[] b) {
        return Math.hypot(a[0] - b[0], a[1] - b[1]);
    }
}
