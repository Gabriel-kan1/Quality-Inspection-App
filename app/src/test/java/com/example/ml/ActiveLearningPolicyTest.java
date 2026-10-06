package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Pins the selection budget, which is easy to get subtly wrong and invisible when it is.
 *
 * The policy decides how many dialogs an operator is shown per inspection and what mix of
 * them; get the arithmetic wrong and either the cap is exceeded (operators learn to mash
 * skip) or spot checks crowd out every real unknown (the tuning set goes back to being
 * biased toward the classifier's own failures, which is the bias spot checks exist to
 * correct).
 *
 * Robolectric is needed only for ReferenceLibrary's android.graphics.Color static block.
 */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = {36})
public class ActiveLearningPolicyTest {

    private static final long SEED = 42L;

    private static ReferenceLibrary libraryWithTwoPrototypes() {
        ReferenceLibrary lib = new ReferenceLibrary();
        lib.addExample("stud", new float[]{1f, 0f, 0f});
        lib.addExample("bolt_head", new float[]{0f, 1f, 0f});
        return lib;
    }

    /** A classification whose best cosine is {@code score}, with a wide runner-up gap. */
    private static ReferenceLibrary.Classification scoringAt(ReferenceLibrary lib, float score) {
        float axis = (float) Math.sqrt(Math.max(0, 1 - score * score));
        return lib.classify(new float[]{score, axis * 0.05f, axis});
    }

    /** An unknown sitting at {@code score}, just under the threshold. */
    private static ReferenceLibrary.Classification unknownAt(ReferenceLibrary lib, float score) {
        return scoringAt(lib, score);
    }

    private static List<ReferenceLibrary.Classification> unknowns(ReferenceLibrary lib, int n) {
        List<ReferenceLibrary.Classification> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(unknownAt(lib, 0.70f - i * 0.001f));
        }
        return out;
    }

    /** A confidently accepted detection, i.e. a spot-check candidate. */
    private static ReferenceLibrary.Classification accepted(ReferenceLibrary lib) {
        ReferenceLibrary.Classification c = lib.classify(new float[]{1f, 0f, 0f});
        assertEquals("fixture must actually be accepted", "stud", c.type);
        assertEquals("fixture must actually be high confidence", "high", c.confidence);
        return c;
    }

    /** A detection the noise gate routed to its ignored type without asking. */
    private static ReferenceLibrary.Classification suppressed(ReferenceLibrary lib) {
        ReferenceLibrary.Classification raw = scoringAt(lib, 0.66f);
        assertEquals("fixture must start as an unknown", ReferenceLibrary.UNKNOWN, raw.type);
        return NoiseGate.asIgnored(raw);
    }

    private static int countOf(List<ActiveLearningPolicy.Candidate> selected,
                               ActiveLearningPolicy.Kind kind) {
        int n = 0;
        for (ActiveLearningPolicy.Candidate c : selected) {
            if (c.kind == kind) n++;
        }
        return n;
    }

    @Test
    public void neverExceedsTheDialogBudget() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>(unknowns(lib, 50));
        all.add(accepted(lib));

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        assertTrue("a 50-dialog gauntlet trains operators to mash skip",
                selected.size() <= ActiveLearningPolicy.MAX_REVIEWS_PER_INSPECTION);
    }

    @Test
    public void spotChecksNeverCrowdOutEveryUnknown() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>(unknowns(lib, 20));
        for (int i = 0; i < 20; i++) {
            all.add(accepted(lib));
        }

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        assertTrue("unknowns are the reason the queue exists",
                countOf(selected, ActiveLearningPolicy.Kind.UNKNOWN) > 0);
        assertEquals(ActiveLearningPolicy.SPOT_CHECKS_PER_INSPECTION,
                countOf(selected, ActiveLearningPolicy.Kind.SPOT_CHECK));
    }

    @Test
    public void samplesAConfidentPredictionToKeepTheTuningSetHonest() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>(unknowns(lib, 3));
        all.add(accepted(lib));

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        assertEquals("without these the tuner only ever sees the classifier's failures",
                1, countOf(selected, ActiveLearningPolicy.Kind.SPOT_CHECK));
    }

    /**
     * The band that used to be invisible to the sample.
     *
     * Spot-check eligibility was gated on {@code "high"} confidence, so any accepted
     * prediction scoring between the similarity threshold and
     * {@link ReferenceLibrary#DEFAULT_HIGH_CONFIDENCE} could never be drawn. Measured over
     * this project's own inspection history that band held 88 of 141 accepted predictions
     * and every one of the three known silent misclassifications — and it had received zero
     * spot checks. The operator found all three by tapping boxes by hand.
     *
     * A sample that cannot reach the population where the errors are is not bias control,
     * it is a second bias. The pool is now every accepted prediction, drawn uniformly.
     */
    @Test
    public void aMediumConfidenceAcceptanceIsStillWorthSpotChecking() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        ReferenceLibrary.Classification c = scoringAt(lib, 0.80f);
        assertEquals("fixture must be accepted", "stud", c.type);
        assertEquals("fixture must sit in the band that used to be exempt",
                "medium", c.confidence);

        List<ReferenceLibrary.Classification> all = new ArrayList<>();
        all.add(c);
        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        assertEquals("a merely-accepted prediction must still be auditable",
                1, countOf(selected, ActiveLearningPolicy.Kind.SPOT_CHECK));
    }

    /**
     * A suppressed detection is never queued as an unknown — that is the whole point — but
     * it is not silently trusted either. One per run comes back as a noise audit, and it is
     * drawn from its own pool so it cannot dilute the spot check on real hardware.
     */
    @Test
    public void aSuppressedDetectionIsAuditedNotQueued() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>();
        for (int i = 0; i < 12; i++) all.add(suppressed(lib));
        all.add(accepted(lib));

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        assertEquals("nothing suppressed may reappear as an unknown prompt",
                0, countOf(selected, ActiveLearningPolicy.Kind.UNKNOWN));
        assertEquals(1, countOf(selected, ActiveLearningPolicy.Kind.NOISE_AUDIT));
        assertEquals("twelve suppressed crops must not crowd out the hardware spot check",
                1, countOf(selected, ActiveLearningPolicy.Kind.SPOT_CHECK));
        assertEquals(2, selected.size());
    }

    @Test
    public void noiseAuditsNeverDisplaceUnknownsOrSpotChecks() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>(unknowns(lib, 20));
        for (int i = 0; i < 5; i++) all.add(accepted(lib));
        for (int i = 0; i < 5; i++) all.add(suppressed(lib));

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        assertTrue(selected.size() <= ActiveLearningPolicy.MAX_REVIEWS_PER_INSPECTION);
        assertTrue(countOf(selected, ActiveLearningPolicy.Kind.UNKNOWN) > 0);
        assertEquals(1, countOf(selected, ActiveLearningPolicy.Kind.SPOT_CHECK));
        assertEquals(1, countOf(selected, ActiveLearningPolicy.Kind.NOISE_AUDIT));
    }

    /** The audit exists to check the gate; with nothing suppressed there is nothing to check. */
    @Test
    public void noSuppressionMeansNoAudit() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>(unknowns(lib, 3));
        all.add(accepted(lib));

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        assertEquals(0, countOf(selected, ActiveLearningPolicy.Kind.NOISE_AUDIT));
    }

    @Test
    public void withNothingConfidentThereIsNothingToSpotCheck() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(unknowns(lib, 3), lib, new Random(SEED));

        assertEquals(0, countOf(selected, ActiveLearningPolicy.Kind.SPOT_CHECK));
        assertEquals(3, selected.size());
    }

    @Test
    public void boundaryCasesAreQueuedAheadOfCropsThatMatchNothing() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>();
        // Resembles nothing in the library: possibly novel, possibly junk.
        all.add(unknownAt(lib, 0.10f));
        // Sitting right on the decision boundary: labelling this one moves it.
        all.add(unknownAt(lib, 0.73f));

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        assertEquals(1, selected.get(0).featureIndex);
        assertTrue(selected.get(0).priority > selected.get(1).priority);
    }

    @Test
    public void nothingDetectedMeansNothingToAsk() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        assertTrue(ActiveLearningPolicy.select(new ArrayList<>(), lib, new Random(SEED)).isEmpty());
    }

    @Test
    public void nullClassificationsAreSkippedRatherThanSelected() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>();
        all.add(null);
        all.add(unknownAt(lib, 0.72f));

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        assertEquals(1, selected.size());
        assertEquals(1, selected.get(0).featureIndex);
    }

    @Test
    public void aUserInitiatedCorrectionIsNeverAutoSelected() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>(unknowns(lib, 5));

        List<ActiveLearningPolicy.Candidate> selected =
                ActiveLearningPolicy.select(all, lib, new Random(SEED));

        for (ActiveLearningPolicy.Candidate c : selected) {
            assertFalse(c.kind == ActiveLearningPolicy.Kind.USER_INITIATED);
        }
        assertEquals(ActiveLearningPolicy.Kind.USER_INITIATED,
                ActiveLearningPolicy.userInitiated(3).kind);
        assertEquals(3, ActiveLearningPolicy.userInitiated(3).featureIndex);
    }

    // ── What the operator is asked about at all ─────────────────────────────

    private static java.util.Set<Integer> askedAbout(List<ActiveLearningPolicy.Candidate> selected) {
        java.util.Set<Integer> out = new java.util.HashSet<>();
        for (ActiveLearningPolicy.Candidate c : selected) out.add(c.featureIndex);
        return out;
    }

    /**
     * Only a critical type with a best match from the floor to the ceiling is put to the
     * operator: a confident call is trusted, and a crop that resembles nothing is not worth
     * naming.
     */
    @Test
    public void onlyBestMatchesBetweenTheFloorAndTheCeilingAreAsked() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>();
        all.add(accepted(lib));                 // 0: scores 1.00, trusted
        all.add(unknownAt(lib, 0.20f));         // 1: below the floor
        all.add(unknownAt(lib, 0.70f));         // 2: borderline unknown
        all.add(scoringAt(lib, 0.80f));         // 3: accepted, under the ceiling

        java.util.Set<Integer> asked = askedAbout(
                ActiveLearningPolicy.select(all, lib, new Random(SEED), type -> true));

        assertFalse("above the ceiling is trusted", asked.contains(0));
        assertFalse("below the floor is not worth naming", asked.contains(1));
        assertTrue("a borderline unknown is still asked about", asked.contains(2));
        assertTrue("an accepted call under the ceiling can still be spot-checked",
                asked.contains(3));
    }

    @Test
    public void advisoryAndIgnoredTypesAreNeverAsked() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>(unknowns(lib, 3));
        all.add(scoringAt(lib, 0.80f));
        // Every fixture's nearest type is stud; make stud a non-critical type.
        assertTrue(ActiveLearningPolicy.select(all, lib, new Random(SEED),
                type -> !"stud".equals(type)).isEmpty());
    }

    /** What the noise gate routed to an ignored type stays there: no audit prompt either. */
    @Test
    public void aDetectionRoutedToAnIgnoredTypeIsNotAudited() {
        ReferenceLibrary lib = libraryWithTwoPrototypes();
        List<ReferenceLibrary.Classification> all = new ArrayList<>();
        for (int i = 0; i < 3; i++) all.add(suppressed(lib));

        List<ActiveLearningPolicy.Candidate> selected = ActiveLearningPolicy.select(all, lib,
                new Random(SEED), type -> !"stud".equals(type));

        assertEquals(0, countOf(selected, ActiveLearningPolicy.Kind.NOISE_AUDIT));
        assertTrue(selected.isEmpty());
    }

    private static ReferenceLibrary.Classification bestMatch(float score) {
        return new ReferenceLibrary.Classification("stud", "stud", score, "medium", null, -1f,
                ReferenceLibrary.Reason.ACCEPTED, new java.util.HashMap<>());
    }

    /** "Below 0.30" and "above 0.85" are what is left out; the two lines themselves are asked. */
    @Test
    public void theFloorAndTheCeilingThemselvesAreAsked() {
        assertTrue(ActiveLearningPolicy.worthAsking(bestMatch(0.30f), type -> true));
        assertFalse(ActiveLearningPolicy.worthAsking(bestMatch(0.299f), type -> true));
        assertTrue(ActiveLearningPolicy.worthAsking(bestMatch(0.85f), type -> true));
        assertFalse(ActiveLearningPolicy.worthAsking(bestMatch(0.851f), type -> true));
    }
}
