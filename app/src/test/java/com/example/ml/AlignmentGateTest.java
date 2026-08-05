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
        float[] sameArea = {SCREWS_TEST, REFERENCE, DEFECTED, SCREWS_THREE, SCREWS_FLASH, SCREWS_TWO};
        for (float similarity : sameArea) {
            assertTrue("similarity " + similarity + " should pass",
                    AlignmentGate.classify(similarity, STEADY).inTolerance);
        }
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
