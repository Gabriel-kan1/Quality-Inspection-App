package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Pins the rule that decides between framing scales — the one part of the search that can be
 * checked without OpenCV, and the part that was wrong.
 *
 * <p>The numbers are the measured correlation curve for {@code whole screws1.jpg} against
 * {@code whole screws2.jpg}: the same plate from the same distance, which the raw peak read as
 * "3.3x too far" because a 30% template's chance-level 0.48 beat the true scale's 0.47.
 */
public class TemplateMatcherRankingTest {

    @Test
    public void anUnchangedFramingBeatsASmallTemplatesChancePeak() {
        double tiny = TemplateMatcher.ranked(0.48, 0.30);
        double unchanged = TemplateMatcher.ranked(0.47, 0.95);
        assertTrue("0.47 at 0.95 must outrank 0.48 at 0.30", unchanged > tiny);
    }

    /** A genuine match at a small scale — the exact crop cases score 0.6 and up — still wins. */
    @Test
    public void aStrongPeakAtASmallScaleStillWins() {
        double crop = TemplateMatcher.ranked(0.65, 0.40);
        double full = TemplateMatcher.ranked(0.45, 1.00);
        assertTrue(crop > full);
    }

    @Test
    public void thePriorIsZeroAtUnchangedFramingAndSymmetricAboutIt() {
        assertEquals(0.52, TemplateMatcher.ranked(0.52, 1.0), 1e-12);
        assertEquals(TemplateMatcher.ranked(0.5, 0.5), TemplateMatcher.ranked(0.5, 2.0), 1e-12);
        assertEquals(0.5 - TemplateMatcher.SCALE_PRIOR * Math.log(2.0),
                TemplateMatcher.ranked(0.5, 0.5), 1e-12);
    }

    /** The constant sits inside the plateau the study found; outside it the trade-off changes. */
    @Test
    public void thePriorIsOnTheMeasuredPlateau() {
        assertTrue(TemplateMatcher.SCALE_PRIOR >= 0.08 && TemplateMatcher.SCALE_PRIOR <= 0.12);
    }
}
