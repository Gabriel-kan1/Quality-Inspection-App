package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Pins the gate's decision rule against the similarities actually measured over the
 * bundled chassis assets, so a future threshold change cannot silently start rejecting
 * the same-area case the gate exists to accept.
 */
public class AlignmentGateTest {

    // Measured with screws.jpg as the reference, using FeatureEmbedder's preprocessing.
    private static final float SCREWS_TEST = 0.747f;   // same area, different hardware
    private static final float REFERENCE = 0.739f;     // same area
    private static final float DEFECTED = 0.732f;      // same area
    private static final float SCREWS_THREE = 0.659f;  // same area
    private static final float SCREWS_FLASH = 0.603f;  // same area, flash
    private static final float SCREWS_TWO = 0.578f;    // same area, further back
    private static final float ZERO = 0.510f;          // different area — closest false positive
    private static final float CORNER = 0.309f;        // different area
    private static final float WHOLE = 0.218f;         // different area

    // Wrong-distance shots of the SAME area, measured against stud1.jpg: similarity alone
    // clears the threshold, and only the scale probe separates them. Gains are
    // (crop-the-frame, crop-the-reference).
    private static final float SCREWS_TWO_VS_STUD1 = 0.717f;   // ~2.5x too far
    private static final float SCREWS_TWO_FAR_GAIN = 0.168f;
    private static final float SCREWS_THREE_VS_STUD1 = 0.676f; // ~2.8x too far
    private static final float SCREWS_THREE_FAR_GAIN = 0.104f;
    private static final float STUD1_VS_SCREWS_TWO = 0.717f;   // the same pair, reference swapped
    private static final float STUD1_CLOSE_GAIN = 0.168f;

    // Correctly framed frames: cropping either side makes the match worse, so both gains
    // are negative. p90 over the labelled pairs is +0.012, well inside the tolerance.
    private static final float STEADY_FAR_GAIN = -0.070f;
    private static final float STEADY_CLOSE_GAIN = -0.087f;

    private static final float STEADY = 1.5f;
    private static final float MOVING = 20f;

    @Test
    public void acceptsTheSameAreaWithDifferentHardware() {
        // The case this gate was rebuilt for: same four holes, swapped and missing fasteners.
        AlignmentGate.Reading r = AlignmentGate.classify(SCREWS_TEST, STEADY);
        assertTrue("screws_test must be accepted as the same area", r.inTolerance);
        assertEquals(AlignmentGate.Fault.NONE, r.fault);
    }

    @Test
    public void acceptsEverySameAreaShotThatIsCloseEnoughToInspect() {
        // SCREWS_TWO and the other wide shots are deliberately absent: they are the same
        // area photographed from too far back, which the scale probe now rejects on
        // purpose. See rejectsTheSameAreaShotFromTooFarAway().
        float[] sameArea = {SCREWS_TEST, REFERENCE, DEFECTED};
        for (float similarity : sameArea) {
            assertTrue("similarity " + similarity + " should pass",
                    AlignmentGate.classify(similarity, STEADY_FAR_GAIN, STEADY_CLOSE_GAIN, STEADY)
                            .inTolerance);
        }
    }

    @Test
    public void rejectsTheSameAreaShotFromTooFarAway() {
        // The complaint this probe exists for: same hardware, twice the distance, and a
        // similarity that clears the threshold on its own.
        AlignmentGate.Reading r = AlignmentGate.classify(
                SCREWS_TWO_VS_STUD1, SCREWS_TWO_FAR_GAIN, -0.103f, STEADY);
        assertFalse("a shot from 2.5x the reference distance must not pass", r.inTolerance);
        assertEquals(AlignmentGate.Fault.TOO_FAR, r.fault);

        AlignmentGate.Reading nearer = AlignmentGate.classify(
                SCREWS_THREE_VS_STUD1, SCREWS_THREE_FAR_GAIN, -0.143f, STEADY);
        assertFalse(nearer.inTolerance);
        assertEquals(AlignmentGate.Fault.TOO_FAR, nearer.fault);
    }

    @Test
    public void rejectsTheSameAreaShotFromTooCloseIn() {
        AlignmentGate.Reading r = AlignmentGate.classify(
                STUD1_VS_SCREWS_TWO, -0.103f, STUD1_CLOSE_GAIN, STEADY);
        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.TOO_CLOSE, r.fault);
    }

    @Test
    public void distanceIsReportedBeforeSteadiness() {
        // A phone held at the wrong distance is steady-wrong; "hold still" would strand
        // the operator there.
        AlignmentGate.Reading r = AlignmentGate.classify(
                SCREWS_TWO_VS_STUD1, SCREWS_TWO_FAR_GAIN, -0.103f, MOVING);
        assertEquals(AlignmentGate.Fault.TOO_FAR, r.fault);
    }

    @Test
    public void aFrameShortOfTheThresholdIsNotBlamedOnDistance() {
        // Below the threshold the probe never runs, so its gains arrive as NaN and must
        // not be read as "within tolerance".
        AlignmentGate.Reading r = AlignmentGate.classify(WHOLE, Float.NaN, Float.NaN, STEADY);
        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.NO_MATCH, r.fault);
    }

    @Test
    public void aWrongDistanceFrameScoresLowerThanItsRawSimilarity() {
        // What the operator sees has to agree with the verdict: a frame rejected for
        // distance cannot keep advertising a 72% match.
        AlignmentGate.Reading r = AlignmentGate.classify(
                SCREWS_TWO_VS_STUD1, SCREWS_TWO_FAR_GAIN, -0.103f, STEADY);
        assertEquals(55, r.matchPercent());

        // A correctly framed frame has negative gains and keeps its raw score.
        AlignmentGate.Reading ok = AlignmentGate.classify(
                SCREWS_TEST, STEADY_FAR_GAIN, STEADY_CLOSE_GAIN, STEADY);
        assertEquals(75, ok.matchPercent());
    }

    @Test
    public void rejectsDifferentAreasOfTheChassis() {
        float[] differentArea = {ZERO, CORNER, WHOLE};
        for (float similarity : differentArea) {
            assertFalse("similarity " + similarity + " should fail",
                    AlignmentGate.classify(similarity, STEADY).inTolerance);
        }
    }

    @Test
    public void aMatchingButMovingFrameDoesNotCountTowardTheHold() {
        AlignmentGate.Reading r = AlignmentGate.classify(SCREWS_TEST, MOVING);
        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.MOVING, r.fault);
    }

    @Test
    public void theFirstFrameIsNotJudgedAsMoving() {
        // NaN motion means there was no previous frame to compare against.
        assertTrue(AlignmentGate.classify(SCREWS_TEST, Float.NaN).inTolerance);
    }

    @Test
    public void aNearMissIsDistinguishedFromLookingAtNothingRelevant() {
        assertEquals(AlignmentGate.Fault.WEAK_MATCH, AlignmentGate.classify(ZERO, STEADY).fault);
        assertEquals(AlignmentGate.Fault.NO_MATCH, AlignmentGate.classify(WHOLE, STEADY).fault);
    }

    @Test
    public void matchPercentIsClampedToADisplayableRange() {
        assertEquals(75, AlignmentGate.classify(0.747f, STEADY).matchPercent());
        assertEquals(0, AlignmentGate.classify(-0.4f, STEADY).matchPercent());
        assertEquals(100, AlignmentGate.classify(1.0f, STEADY).matchPercent());
    }
}
