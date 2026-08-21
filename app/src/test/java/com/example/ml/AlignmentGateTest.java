package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Pins the gate's decision rule: which of the four things that can be wrong with a frame it
 * reports, in which order, and what it tells the operator to do about it.
 *
 * <p>Area similarities are the ones measured over the bundled chassis assets with
 * {@code screws.jpg} as the reference. Framing figures are supplied directly as
 * {@link TemplateMatcher.Match} values so a test states only what it is about — the
 * matcher's own accuracy is a separate question, measured in {@code python/} and recorded in
 * its class notes.
 */
public class AlignmentGateTest {

    // Measured with screws.jpg as the reference, using FeatureEmbedder's preprocessing.
    private static final float SCREWS_TEST = 0.747f;   // same area, different hardware
    private static final float REFERENCE = 0.739f;     // same area
    private static final float DEFECTED = 0.732f;      // same area
    private static final float ZERO = 0.510f;          // different area — closest false positive
    private static final float WHOLE = 0.218f;         // different area

    private static final float STEADY = 1.5f;
    private static final float MOVING = 20f;

    /** A frame framed the way the reference was: right size, centred, solid correlation. */
    private static TemplateMatcher.Match wellFramed() {
        return new TemplateMatcher.Match(0.45f, 1.00f, 0f, 0f);
    }

    private static TemplateMatcher.Match framed(float scale, float offX, float offY) {
        return new TemplateMatcher.Match(0.45f, scale, offX, offY);
    }

    private static AlignmentGate.Reading judge(float similarity, TemplateMatcher.Match m, float motion) {
        return AlignmentGate.classify(similarity, m, motion, -1L);
    }

    @Test
    public void acceptsTheSameAreaCorrectlyFramed() {
        // The case the gate exists for: same four holes, swapped and missing fasteners,
        // photographed the way the reference was.
        AlignmentGate.Reading r = judge(SCREWS_TEST, wellFramed(), STEADY);
        assertTrue(r.inTolerance);
        assertEquals(AlignmentGate.Fault.NONE, r.fault);
    }

    @Test
    public void acceptsEverySameAreaShotThatIsFramedLikeTheReference() {
        for (float similarity : new float[]{SCREWS_TEST, REFERENCE, DEFECTED}) {
            assertTrue("similarity " + similarity + " should pass",
                    judge(similarity, wellFramed(), STEADY).inTolerance);
        }
    }

    /**
     * The complaint this was rebuilt for.
     *
     * <p>{@code defected.jpg} is the reference's own fastener cluster shot from far closer,
     * so it scores as the same area and passed the old gate at 78%. Two of the six expected
     * fasteners were never in the photograph, and the inspection then reported them missing
     * and failed a good part. The frame covering a fraction of the reference is exactly what
     * the framing ratio measures.
     */
    @Test
    public void rejectsAZoomedInShotOfTheReferenceArea() {
        AlignmentGate.Reading r = judge(DEFECTED, framed(0.57f, 0f, 0f), STEADY);
        assertFalse("a frame showing 57% of the reference must not pass", r.inTolerance);
        assertEquals(AlignmentGate.Fault.TOO_CLOSE, r.fault);
        assertTrue("the hint must say how far out it is, was: " + r.hint,
                r.hint.contains("57%"));
    }

    @Test
    public void rejectsAShotFromTooFarBack() {
        AlignmentGate.Reading r = judge(SCREWS_TEST, framed(2.0f, 0f, 0f), STEADY);
        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.TOO_FAR, r.fault);
    }

    @Test
    public void aimIsReportedWithADirection() {
        assertEquals(AlignmentGate.Fault.AIM_RIGHT,
                judge(SCREWS_TEST, framed(1.0f, 0.30f, 0f), STEADY).fault);
        assertEquals(AlignmentGate.Fault.AIM_LEFT,
                judge(SCREWS_TEST, framed(1.0f, -0.30f, 0f), STEADY).fault);
        assertEquals(AlignmentGate.Fault.AIM_DOWN,
                judge(SCREWS_TEST, framed(1.0f, 0f, 0.30f), STEADY).fault);
        assertEquals(AlignmentGate.Fault.AIM_UP,
                judge(SCREWS_TEST, framed(1.0f, 0f, -0.30f), STEADY).fault);
    }

    @Test
    public void theLargerAimErrorIsTheOneReported() {
        // Only one instruction is given at a time, so it has to be the one that helps most.
        assertEquals(AlignmentGate.Fault.AIM_UP,
                judge(SCREWS_TEST, framed(1.0f, 0.14f, -0.30f), STEADY).fault);
        assertEquals(AlignmentGate.Fault.AIM_RIGHT,
                judge(SCREWS_TEST, framed(1.0f, 0.30f, -0.14f), STEADY).fault);
    }

    @Test
    public void aSmallAimErrorIsTolerated() {
        assertTrue(judge(SCREWS_TEST, framed(1.0f, 0.05f, -0.05f), STEADY).inTolerance);
    }

    @Test
    public void distanceIsReportedBeforeAim() {
        // Distance decides whether the part is even fully in shot; correcting aim first
        // would walk the operator around a frame that can never cover it.
        AlignmentGate.Reading r = judge(SCREWS_TEST, framed(0.5f, 0.40f, 0f), STEADY);
        assertEquals(AlignmentGate.Fault.TOO_CLOSE, r.fault);
    }

    @Test
    public void framingIsReportedBeforeSteadiness() {
        // A phone held at the wrong distance is steady-wrong; "hold still" would strand
        // the operator there.
        assertEquals(AlignmentGate.Fault.TOO_CLOSE,
                judge(SCREWS_TEST, framed(0.5f, 0f, 0f), MOVING).fault);
        assertEquals(AlignmentGate.Fault.AIM_RIGHT,
                judge(SCREWS_TEST, framed(1.0f, 0.30f, 0f), MOVING).fault);
    }

    @Test
    public void aMatchingWellFramedButMovingFrameDoesNotCountTowardTheHold() {
        AlignmentGate.Reading r = judge(SCREWS_TEST, wellFramed(), MOVING);
        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.MOVING, r.fault);
    }

    @Test
    public void theFirstFrameIsNotJudgedAsMoving() {
        // NaN motion means there was no previous frame to compare against.
        assertTrue(judge(SCREWS_TEST, wellFramed(), Float.NaN).inTolerance);
    }

    @Test
    public void rejectsDifferentAreasOfTheChassis() {
        // Framing is irrelevant when the frame is not the right area at all — and note the
        // framing here is perfect, so nothing but the area test can be rejecting it.
        for (float similarity : new float[]{ZERO, WHOLE}) {
            assertFalse("similarity " + similarity + " should fail",
                    judge(similarity, wellFramed(), STEADY).inTolerance);
        }
    }

    @Test
    public void aNearMissIsDistinguishedFromLookingAtNothingRelevant() {
        assertEquals(AlignmentGate.Fault.WEAK_MATCH, judge(ZERO, null, STEADY).fault);
        assertEquals(AlignmentGate.Fault.NO_MATCH, judge(WHOLE, null, STEADY).fault);
    }

    /**
     * The safety property of the rebuild: no framing measurement, no pass.
     *
     * <p>The search can fail — OpenCV missing, a reference whose gradients could not be
     * prepared, a correlation peak indistinguishable from noise. Treating that as "framing
     * is probably fine" would let through exactly the capture the search exists to stop,
     * silently, on the devices where it is least able to help.
     */
    @Test
    public void anAreaMatchWithNoFramingMeasurementMustNotPass() {
        AlignmentGate.Reading r = judge(SCREWS_TEST, null, STEADY);
        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.NO_FRAMING, r.fault);
    }

    @Test
    public void aCorrelationPeakInTheNoiseIsNotAFramingMeasurement() {
        AlignmentGate.Reading r = judge(SCREWS_TEST, new TemplateMatcher.Match(0.05f, 1.0f, 0f, 0f), STEADY);
        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.NO_FRAMING, r.fault);
    }

    /**
     * Pins the area threshold, because raising it makes the tests above fail for reasons that
     * read like arithmetic rather than like the actual objection.
     *
     * <p>Over the 252 labelled pairs in {@code python/scale_probe_study.py}, 0.70 is the last
     * threshold that costs no true accepts. Scored against the RANSAC-verified pairs from
     * {@code python/pair_ground_truth.py}, recall falls from 17/19 to 15/19 by 0.74, and the
     * pairs lost are proven same-area by several hundred inliers. Re-run both before changing it.
     */
    @Test
    public void theAreaThresholdStaysWhereTheStudiesPutIt() {
        assertFalse("must not be looser than 0.70 — see python/scale_probe_study.py",
                judge(0.699f, wellFramed(), STEADY).inTolerance);
        assertTrue("must not be tighter than 0.70 — see python/pair_ground_truth.py",
                judge(0.701f, wellFramed(), STEADY).inTolerance);
    }

    /**
     * Each hint carries the figure that explains its own fault, and no other.
     *
     * <p>The screen used to append the area score to every rejection, so a frame turned away
     * for covering 60% of the part was captioned "TOO CLOSE — MOVE BACK ... — 78% match".
     * Both numbers were true and the pairing read as a contradiction: the operator is being
     * told the match is good and that the frame is rejected, with nothing linking the number
     * to the reason. The area score is what they climb toward the threshold, so it belongs on
     * an area miss and is noise anywhere else.
     */
    @Test
    public void aHintCarriesOnlyTheFigureThatExplainsItsOwnFault() {
        String weak = judge(ZERO, null, STEADY).hint;
        assertTrue("an area miss states the area score, was: " + weak, weak.contains("51%"));

        String tooClose = judge(DEFECTED, framed(0.57f, 0f, 0f), STEADY).hint;
        assertTrue("a framing fault states its own coverage", tooClose.contains("57%"));
        assertFalse("and must not also advertise the area score, was: " + tooClose,
                tooClose.contains("73%"));

        String aim = judge(SCREWS_TEST, framed(1.0f, 0.30f, 0f), STEADY).hint;
        assertTrue("an aim fault states how far off it is", aim.contains("30%"));
        assertFalse("and not the area score, was: " + aim, aim.contains("75%"));

        String moving = judge(SCREWS_TEST, wellFramed(), MOVING).hint;
        assertFalse("steadiness needs no percentage at all, was: " + moving,
                moving.contains("%"));
    }

    @Test
    public void matchPercentIsClampedToADisplayableRange() {
        assertEquals(75, judge(0.747f, wellFramed(), STEADY).matchPercent());
        assertEquals(0, judge(-0.4f, null, STEADY).matchPercent());
        assertEquals(100, judge(1.0f, wellFramed(), STEADY).matchPercent());
    }

    @Test
    public void theTemplateScoreIsCarriedSeparatelyFromTheAreaScore() {
        // Two different measurements answering two different questions; collapsing them into
        // one number is what made the old gate unable to say which was wrong.
        AlignmentGate.Reading r = judge(0.747f, new TemplateMatcher.Match(0.42f, 1.0f, 0f, 0f), STEADY);
        assertEquals(75, r.matchPercent());
        assertEquals(42, r.templatePercent());
    }
}
