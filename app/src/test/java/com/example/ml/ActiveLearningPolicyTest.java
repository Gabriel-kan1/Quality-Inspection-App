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

    /** An unknown sitting at {@code score}, just under the threshold. */
    private static ReferenceLibrary.Classification unknownAt(ReferenceLibrary lib, float score) {
        float axis = (float) Math.sqrt(Math.max(0, 1 - score * score));
        return lib.classify(new float[]{score, axis * 0.05f, axis});
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
}
