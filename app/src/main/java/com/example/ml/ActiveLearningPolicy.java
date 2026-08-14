package com.example.ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Chooses which detections are worth interrupting the operator for.
 *
 * Prompting on every "unknown" doesn't scale: as the library grows, the leftover
 * unknowns are increasingly junk contours rather than genuine unlabelled hardware,
 * and a 30-dialog gauntlet trains operators to mash "skip". Instead we rank by how
 * much a human answer would actually teach the classifier and cap the prompts.
 *
 * Two kinds of candidate get selected:
 *
 * 1. UNKNOWNS, ranked by informativeness. A crop that landed just under the
 *    similarity threshold, or that was split between two types, sits right on the
 *    decision boundary — labelling it moves the boundary. A crop that scored near
 *    zero against everything is either a novel feature type or noise; still worth
 *    asking about, but only after the boundary cases.
 *
 * 2. SPOT CHECKS: a small uniform random sample of *accepted* predictions.
 *    These matter more than they look. If the only labels we ever collect come from
 *    crops the classifier already called "unknown", the tuning set is systematically
 *    biased toward its own failures, and {@link ThresholdTuner} will happily drive
 *    thresholds to zero to "fix" a population that contains no true accepts. Spot
 *    checks put correctly-classified examples back in the sample so the fit is honest.
 *
 *    <h3>Why the pool is every accepted prediction, not the confident ones</h3>
 *    It used to be gated on {@code "high"} confidence, i.e. a score above
 *    {@link ReferenceLibrary#DEFAULT_HIGH_CONFIDENCE}. That defeated the purpose. The
 *    point of the sample is to be <em>unbiased</em> with respect to what the classifier
 *    thinks, and drawing only from the safest quartile is no less biased than drawing
 *    only from the failures — merely biased the other way, and toward the crops with
 *    least to teach.
 *
 *    Measured on this project's own inspection history: of 141 accepted predictions on
 *    the shipping encoder, 88 sat in the 0.74-0.88 band and received <em>zero</em> spot
 *    checks between them. Every one of the three known silent misclassifications lived
 *    in that band, and every one was found by the operator tapping the box by hand — the
 *    review loop surfaced none of them. A sample that cannot reach the population where
 *    the errors are is not bias control. See {@code python/margin_study.py}.
 */
public final class ActiveLearningPolicy {

    /** Hard cap on dialogs per inspection, so the loop stays usable as detections grow. */
    public static int MAX_REVIEWS_PER_INSPECTION = 8;
    /** How many already-accepted predictions to sample for confirmation (bias control). */
    public static int SPOT_CHECKS_PER_INSPECTION = 1;
    /**
     * Scores below this are treated as "resembles nothing in the library" rather than
     * "near the boundary", and are queued behind the genuinely borderline cases.
     *
     * Anchored to the encoder's mean similarity between crops of *different* types
     * (measured 0.47 for {@link FeatureEmbedder#ENCODER_ID}). Scoring no better than an
     * arbitrary mismatched pair is what "resembles nothing" actually means, so the
     * constant is derived from the embedding space rather than picked by feel.
     *
     * This was 0.45 under the previous CLIP encoder, which never scored any crop below
     * 0.86 — the branch it guards was unreachable, so every unknown was ranked as a
     * boundary case and novel-feature crops competed for review slots with junk
     * contours. Re-derive this with python/compare_encoders.py if the encoder changes;
     * an inter-class mean is encoder-specific, not a universal constant.
     */
    public static float NOISE_FLOOR = 0.50f;

    public enum Kind {
        UNKNOWN,
        SPOT_CHECK,
        /** Operator tapped a detection on the overlay to correct it; never auto-selected. */
        USER_INITIATED
    }

    /** Builds a candidate for a correction the operator started themselves. */
    public static Candidate userInitiated(int featureIndex) {
        return new Candidate(featureIndex, Kind.USER_INITIATED, 0f);
    }

    public static final class Candidate {
        public final int featureIndex;
        public final Kind kind;
        public final float priority; // higher = more informative

        Candidate(int featureIndex, Kind kind, float priority) {
            this.featureIndex = featureIndex;
            this.kind = kind;
            this.priority = priority;
        }
    }

    private ActiveLearningPolicy() {}

    /**
     * Selects which detections to ask about, most informative first.
     *
     * @param classifications one entry per detected feature, index-aligned with the caller's list.
     * @param library the library that produced them (supplies the live thresholds).
     * @param seed randomness source for spot-check sampling.
     */
    public static List<Candidate> select(List<ReferenceLibrary.Classification> classifications,
                                         ReferenceLibrary library, Random seed) {
        List<Candidate> unknowns = new ArrayList<>();
        List<Integer> accepted = new ArrayList<>();

        for (int i = 0; i < classifications.size(); i++) {
            ReferenceLibrary.Classification c = classifications.get(i);
            if (c == null) continue;
            if (ReferenceLibrary.UNKNOWN.equals(c.type)) {
                unknowns.add(new Candidate(i, Kind.UNKNOWN, informativeness(c, library)));
            } else {
                // Every accepted prediction, at any confidence. Filtering to "high" here made
                // the spot check unable to see the band the errors actually occupy.
                accepted.add(i);
            }
        }

        Collections.sort(unknowns, (a, b) -> Float.compare(b.priority, a.priority));

        List<Candidate> selected = new ArrayList<>();
        int budget = Math.max(0, MAX_REVIEWS_PER_INSPECTION);

        // Reserve room for spot checks, but never let them crowd out real unknowns entirely.
        int spotCheckSlots = Math.min(SPOT_CHECKS_PER_INSPECTION, Math.max(0, budget - 1));
        if (accepted.isEmpty()) spotCheckSlots = 0;

        for (Candidate c : unknowns) {
            if (selected.size() >= budget - spotCheckSlots) break;
            selected.add(c);
        }

        Collections.shuffle(accepted, seed);
        for (int i = 0; i < spotCheckSlots && i < accepted.size(); i++) {
            selected.add(new Candidate(accepted.get(i), Kind.SPOT_CHECK, 0f));
        }

        return selected;
    }

    /**
     * How much we expect a human label here to move the classifier. Highest for crops
     * sitting right on a decision boundary, lowest for crops that matched nothing.
     */
    private static float informativeness(ReferenceLibrary.Classification c, ReferenceLibrary library) {
        if (c.reason == ReferenceLibrary.Reason.NO_LIBRARY) {
            // Nothing to be near the boundary of yet — every label is equally foundational.
            return 1f;
        }
        if (c.reason == ReferenceLibrary.Reason.AMBIGUOUS) {
            // Two prototypes are fighting over this crop; resolving it separates them.
            // The tighter the contest, the more a label helps.
            float gap = Math.max(0f, c.margin());
            return 2f - gap;
        }
        // BELOW_THRESHOLD: closer to the threshold = closer to the boundary = more useful.
        if (c.score < NOISE_FLOOR) {
            // Matches nothing known. Possibly a novel type worth naming, possibly junk —
            // ask, but only once the borderline cases are handled.
            return c.score * 0.1f;
        }
        float distance = library.getSimilarityThreshold() - c.score;
        return 1f - Math.max(0f, distance);
    }
}
