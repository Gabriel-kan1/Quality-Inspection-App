package com.example.ml;

/**
 * Every feature's separation from every other feature, held as a matrix.
 *
 * <h3>Why separations rather than positions</h3>
 * A position is only meaningful relative to the frame it was measured in, and the part is
 * photographed freehand — so a fastener at (0.31, 0.44) in the reference and (0.52, 0.29) in
 * the inspection may well be the same physical thing. Nothing can be compared until the
 * transform between the two views is recovered, which is the job {@link LayoutMatcher} does.
 *
 * The distance between two features has no such dependency. Translate the part, rotate it, and
 * the separations are unchanged; photograph it closer and they all scale by one common factor.
 * So the <em>network</em> of separations is a description of the part itself rather than of the
 * photograph, and two features can be tested for mutual consistency before any transform is
 * known at all.
 *
 * <h3>What that buys the positional check</h3>
 * A correspondence set can be grown by agreement: given two features provisionally identified
 * with two slots, a third is admitted only if its separation from <em>both</em> already matches
 * what the layout says it should be, at the scale the first two imply. Errors do not accumulate
 * — each new member is checked against every existing one, so a detection that happens to sit
 * plausibly near one member but not the others is refused rather than absorbed.
 *
 * That is a strictly stronger test than "does this point land inside a tolerance of where a
 * two-point transform put the slot", because a two-point transform is exact by construction:
 * it inherits the noise of its two seed points and carries that error into every prediction it
 * makes. Growing by mutual agreement instead, and only then fitting a transform over the whole
 * agreed set, keeps the seed's noise from setting the answer.
 *
 * <h3>What it deliberately does not do</h3>
 * Separations are invariant to <em>reflection</em> as well as to rotation, so a network alone
 * cannot tell a part from its mirror image. It is also exactly as ambiguous as raw positions
 * under a part's own symmetry: four bolt heads on a square have the same distance matrix
 * however they are relabelled by a quarter turn. Neither is a defect here, because the network
 * is used only to propose a correspondence — {@link LayoutMatcher} then fits a <em>proper</em>
 * similarity transform (rotation and scale, never reflection) over it and verifies the result,
 * which is what closes the mirror case. The symmetry limit is genuine and documented there.
 *
 * Pure geometry over {@code float[2]} points: no Android types, no framework dependency, so
 * every claim above can be exercised directly in a unit test.
 */
public final class DistanceNetwork {

    private final float[][] points;
    /** Symmetric, zero on the diagonal. Held in full rather than triangular — n is small. */
    private final double[][] d;
    private final double spread;

    private DistanceNetwork(float[][] points) {
        this.points = points;
        int n = points.length;
        this.d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double v = Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]);
                d[i][j] = v;
                d[j][i] = v;
            }
        }
        this.spread = rmsFromCentroid(points);
    }

    /** Builds the network over {@code points}, each {@code {x, y}}. Never null, may be empty. */
    public static DistanceNetwork over(float[][] points) {
        return new DistanceNetwork(points == null ? new float[0][] : points);
    }

    public int size() {
        return points.length;
    }

    public float[] point(int i) {
        return points[i];
    }

    /** Separation between two members, in whatever units the points were given in. */
    public double between(int i, int j) {
        return d[i][j];
    }

    /**
     * RMS distance from the centroid — the point set's characteristic size, and the unit every
     * tolerance in the positional check is expressed as a fraction of.
     *
     * RMS rather than the bounding box: a single outlying detection stretches a box without
     * saying anything about how big the part is, whereas it moves an RMS by roughly its own
     * share of the total.
     */
    public double spread() {
        return spread;
    }

    private static double rmsFromCentroid(float[][] pts) {
        if (pts.length == 0) return 0;
        double cx = 0, cy = 0;
        for (float[] p : pts) {
            cx += p[0];
            cy += p[1];
        }
        cx /= pts.length;
        cy /= pts.length;
        double sum = 0;
        for (float[] p : pts) {
            sum += (p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy);
        }
        return Math.sqrt(sum / pts.length);
    }

    /**
     * A set of expected-node → observed-node correspondences that all agree with one another
     * on the network's evidence, together with the scale and the worst disagreement inside it.
     */
    public static final class Consensus {
        /** Observed index identified with each expected node, or -1 where none was. */
        public final int[] observedForNode;
        /** How many nodes were accounted for. */
        public final int size;
        /** Observed units per expected unit, as implied by the seed pair. */
        public final double scale;
        /**
         * The largest disagreement between an observed separation and the expected one, in
         * observed units. Bounded above by the tolerance the set was grown at, so it is a
         * measure of how comfortably the set agreed rather than merely that it did.
         */
        public final double distortion;

        Consensus(int[] observedForNode, int size, double scale, double distortion) {
            this.observedForNode = observedForNode;
            this.size = size;
            this.scale = scale;
            this.distortion = distortion;
        }
    }

    /**
     * Grows correspondence sets, reusing one scratch buffer across every attempt.
     *
     * A matcher is created once per registration and asked for many candidate sets. Allocating
     * the running-cost table inside the growth would churn an {@code expected × observed} array
     * of doubles per seed, hundreds of times per inspection, on a device where bitmap pressure
     * already makes allocation the thing to watch.
     */
    public static final class Matcher {
        private final DistanceNetwork expected;
        private final DistanceNetwork observed;
        /**
         * Running worst disagreement for each candidate pairing against every member accepted
         * so far. Maintained incrementally: admitting a member folds one comparison into each
         * cell, which keeps growth linear in the set size rather than quadratic.
         */
        private final double[][] cost;
        private final boolean[] nodeUsed;
        private final boolean[] observedUsed;
        private final int[] assignment;

        public Matcher(DistanceNetwork expected, DistanceNetwork observed) {
            this.expected = expected;
            this.observed = observed;
            this.cost = new double[expected.size()][observed.size()];
            this.nodeUsed = new boolean[expected.size()];
            this.observedUsed = new boolean[observed.size()];
            this.assignment = new int[expected.size()];
        }

        /**
         * Builds the largest set of correspondences that all agree, starting from two taken as
         * given.
         *
         * @param nodeA,  obsA the first assumed correspondence
         * @param nodeB,  obsB the second, which fixes the scale
         * @param tolerance how far an observed separation may sit from its expected value, in
         *                  observed units, before the pairing is refused
         * @return the grown set, or null when the seed itself is degenerate
         */
        public Consensus grow(int nodeA, int obsA, int nodeB, int obsB, double tolerance) {
            double base = expected.between(nodeA, nodeB);
            if (!(base > 1e-9)) return null;
            double scale = observed.between(obsA, obsB) / base;
            if (!(scale > 1e-9) || !isFinite(scale)) return null;

            java.util.Arrays.fill(nodeUsed, false);
            java.util.Arrays.fill(observedUsed, false);
            java.util.Arrays.fill(assignment, -1);
            for (double[] row : cost) java.util.Arrays.fill(row, 0.0);

            admit(nodeA, obsA, scale);
            admit(nodeB, obsB, scale);
            int size = 2;
            double distortion = 0;

            // Greedy growth: repeatedly take the pairing that disagrees least with everything
            // already accepted, and stop when nothing left agrees at all.
            //
            // Greedy rather than an exhaustive search for the largest agreeing set: that is the
            // maximum-clique problem and is not something to run per seed on a phone. The cost
            // of greed is only that an unusual arrangement might yield a slightly smaller set
            // than the best possible, which shows up as a refusal to register rather than as a
            // wrong answer — and the seed loop tries many starting points anyway.
            while (true) {
                int bestNode = -1, bestObs = -1;
                double bestCost = tolerance;
                for (int k = 0; k < expected.size(); k++) {
                    if (nodeUsed[k]) continue;
                    double[] row = cost[k];
                    for (int r = 0; r < observed.size(); r++) {
                        if (observedUsed[r]) continue;
                        // Strictly less, scanned in a fixed order, so ties resolve to the
                        // lowest indices and the whole registration stays reproducible.
                        if (row[r] < bestCost) {
                            bestCost = row[r];
                            bestNode = k;
                            bestObs = r;
                        }
                    }
                }
                if (bestNode < 0) break;
                distortion = Math.max(distortion, bestCost);
                admit(bestNode, bestObs, scale);
                size++;
            }
            return new Consensus(assignment.clone(), size, scale, distortion);
        }

        /**
         * Accepts one correspondence and folds it into every remaining candidate's running cost.
         *
         * The cost kept is the <em>worst</em> disagreement against any accepted member, not the
         * average. A candidate that matches four members well and one badly is not a good fit
         * that happens to have an outlier — under a similarity transform every separation must
         * hold at once, so one bad separation is proof the pairing is wrong. Averaging would let
         * a large agreeing set vouch for a member it plainly contradicts.
         */
        private void admit(int node, int obs, double scale) {
            nodeUsed[node] = true;
            observedUsed[obs] = true;
            assignment[node] = obs;
            for (int k = 0; k < expected.size(); k++) {
                if (nodeUsed[k]) continue;
                double wanted = scale * expected.between(k, node);
                double[] row = cost[k];
                for (int r = 0; r < observed.size(); r++) {
                    if (observedUsed[r]) continue;
                    row[r] = Math.max(row[r], Math.abs(observed.between(r, obs) - wanted));
                }
            }
        }
    }

    private static boolean isFinite(double v) {
        return !Double.isNaN(v) && !Double.isInfinite(v);
    }
}
