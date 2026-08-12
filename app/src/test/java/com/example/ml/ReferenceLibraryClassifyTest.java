package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

/**
 * Pins the branches of {@link ReferenceLibrary#classify}, which is where a detection
 * becomes a label the operator is shown and the tuner later fits against.
 *
 * Vectors here are built by hand at chosen angles rather than sampled from the encoder,
 * so each test states a similarity outright instead of hoping a real crop lands near one.
 *
 * Robolectric is needed only because the class touches android.graphics.Color in a static
 * initialiser for its display palette.
 */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = {36})
public class ReferenceLibraryClassifyTest {

    private static final float[] STUD_AXIS = {1f, 0f, 0f};
    private static final float[] BOLT_AXIS = {0f, 1f, 0f};

    /**
     * A query scoring exactly {@code toStud} against the stud prototype and {@code toBolt}
     * against the bolt_head one.
     *
     * The two prototypes are orthogonal, so the first two components *are* the similarities
     * and the third just takes up the slack to unit length. Placing a competitor "far from
     * the stud axis" instead does not work: a prototype 78 degrees off the axis is nearer to
     * a query at 53 degrees than the axis itself is, so the intended runner-up quietly wins.
     * Requires toStud^2 + toBolt^2 <= 1.
     */
    private static float[] query(double toStud, double toBolt) {
        double slack = 1 - toStud * toStud - toBolt * toBolt;
        if (slack < 0) throw new IllegalArgumentException("similarities exceed unit length");
        return new float[]{(float) toStud, (float) toBolt, (float) Math.sqrt(slack)};
    }

    /** A unit vector in the x-y plane at {@code radians} from the stud axis. */
    private static float[] atAngle(double radians) {
        return new float[]{(float) Math.cos(radians), (float) Math.sin(radians), 0f};
    }

    /** Library with a stud prototype and, when {@code withCompetitor}, an orthogonal bolt_head. */
    private static ReferenceLibrary library(boolean withCompetitor) {
        ReferenceLibrary lib = new ReferenceLibrary();
        lib.addExample("stud", STUD_AXIS.clone());
        if (withCompetitor) {
            lib.addExample("bolt_head", BOLT_AXIS.clone());
        }
        return lib;
    }

    @Test
    public void anEmptyLibraryClassifiesNothing() {
        ReferenceLibrary.Classification c = new ReferenceLibrary().classify(STUD_AXIS.clone());

        assertEquals(ReferenceLibrary.UNKNOWN, c.type);
        assertEquals(ReferenceLibrary.Reason.NO_LIBRARY, c.reason);
        assertEquals("none", c.closestMatch);
    }

    @Test
    public void aNullEmbeddingIsNotAClassification() {
        assertEquals(ReferenceLibrary.Reason.NO_LIBRARY, library(true).classify(null).reason);
    }

    @Test
    public void aConfidentMatchIsAccepted() {
        ReferenceLibrary.Classification c = library(true).classify(query(0.95, 0.20));

        assertEquals("stud", c.type);
        assertEquals(ReferenceLibrary.Reason.ACCEPTED, c.reason);
        assertEquals("high", c.confidence);
    }

    @Test
    public void aScoreUnderTheThresholdIsRejectedButKeepsItsBestGuess() {
        ReferenceLibrary.Classification c = library(true).classify(query(0.60, 0.20));

        assertEquals(ReferenceLibrary.UNKNOWN, c.type);
        assertEquals(ReferenceLibrary.Reason.BELOW_THRESHOLD, c.reason);
        // The winning prototype survives on closestMatch — this is what lets ThresholdTuner
        // replay "what if the threshold had been lower" against the logged event.
        assertEquals("stud", c.closestMatch);
        assertTrue(c.score < ReferenceLibrary.DEFAULT_SIMILARITY_THRESHOLD);
    }

    @Test
    public void twoTypesTooCloseToCallAreReportedAsUnknown() {
        // Two prototypes 11 degrees apart, queried from just off the bisector: both score
        // well above the threshold and within DEFAULT_MARGIN_THRESHOLD of each other.
        double separation = Math.acos(0.98);
        ReferenceLibrary lib = new ReferenceLibrary();
        lib.addExample("stud", atAngle(0));
        lib.addExample("bolt_head", atAngle(separation));

        ReferenceLibrary.Classification c = lib.classify(atAngle(separation * 0.45));

        assertTrue("both must clear the threshold for ambiguity to be the deciding factor",
                c.score > ReferenceLibrary.DEFAULT_SIMILARITY_THRESHOLD);
        assertTrue(c.margin() < ReferenceLibrary.DEFAULT_MARGIN_THRESHOLD);
        assertEquals(ReferenceLibrary.UNKNOWN, c.type);
        assertEquals(ReferenceLibrary.Reason.AMBIGUOUS, c.reason);
        assertEquals("low", c.confidence);
        assertNotEquals(null, c.runnerUp);
    }

    @Test
    public void aGapSittingExactlyOnTheMarginIsATieNotAPass() {
        // The inclusive comparison at the margin: a 0.93-vs-0.92 pair used to clear a 0.01
        // margin and commit silently. Queried from the exact bisector, so the gap is zero,
        // and checked against a zero margin — the boundary case a strict "<" would pass.
        double separation = Math.acos(0.5);
        ReferenceLibrary lib = new ReferenceLibrary();
        lib.addExample("stud", atAngle(0));
        lib.addExample("bolt_head", atAngle(separation));
        lib.setThresholds(0.5f, 0f);

        ReferenceLibrary.Classification tie = lib.classify(atAngle(separation / 2));

        assertEquals(0f, tie.margin(), 1e-4f);
        assertEquals(ReferenceLibrary.Reason.AMBIGUOUS, tie.reason);
    }

    @Test
    public void aSoleTypeHasNoRunnerUpToCompeteWith() {
        ReferenceLibrary.Classification c = library(false).classify(query(0.95, 0.20));

        assertEquals("stud", c.type);
        assertEquals(ReferenceLibrary.Reason.ACCEPTED, c.reason);
        assertEquals(null, c.runnerUp);
        assertEquals("an absent runner-up cannot make a match ambiguous",
                Float.MAX_VALUE, c.margin(), 0f);
    }

    @Test
    public void tunedThresholdsChangeTheVerdictForTheSameCrop() {
        // The whole point of ThresholdTuner: identical embedding, different decision.
        float[] borderline = query(0.70, 0.20);

        ReferenceLibrary defaults = library(true);
        assertEquals(ReferenceLibrary.UNKNOWN, defaults.classify(borderline).type);

        ReferenceLibrary tuned = library(true);
        tuned.setThresholds(0.65f, ReferenceLibrary.DEFAULT_MARGIN_THRESHOLD);
        assertEquals("stud", tuned.classify(borderline).type);
    }

    @Test
    public void everyPrototypeIsScoredSoTheUiCanShowTheFullPicture() {
        ReferenceLibrary.Classification c = library(true).classify(query(0.95, 0.20));

        assertEquals(2, c.scores.size());
        assertTrue(c.scores.containsKey("stud"));
        assertTrue(c.scores.containsKey("bolt_head"));
    }
}
