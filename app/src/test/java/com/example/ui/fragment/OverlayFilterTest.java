package com.example.ui.fragment;

import static com.example.ml.FeatureTypeRegistry.Severity.ADVISORY;
import static com.example.ml.FeatureTypeRegistry.Severity.CRITICAL;
import static com.example.ml.FeatureTypeRegistry.Severity.IGNORED;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.example.ui.fragment.ResultsFragment.OverlayFilter;

import org.junit.Test;

/**
 * The spatial overlay's views, narrowest first: critical types only, then advisory too, then
 * ignored too, each hiding what resembles nothing in the library (best match under 0.30); then
 * a last view showing only what those three hide.
 */
public class OverlayFilterTest {

    @Test
    public void criticalOnlyKeepsCriticalTypes() {
        assertTrue(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL, CRITICAL, 0.72f, false));
        assertFalse(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL, ADVISORY, 0.9f, false));
        assertFalse(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL, IGNORED, 0.9f, false));
    }

    @Test
    public void eachViewAddsOneSeverity() {
        assertTrue(ResultsFragment.shownOnOverlay(
                OverlayFilter.CRITICAL_ADVISORY, CRITICAL, 0.5f, false));
        assertTrue(ResultsFragment.shownOnOverlay(
                OverlayFilter.CRITICAL_ADVISORY, ADVISORY, 0.5f, false));
        assertFalse(ResultsFragment.shownOnOverlay(
                OverlayFilter.CRITICAL_ADVISORY, IGNORED, 0.5f, false));

        assertTrue(ResultsFragment.shownOnOverlay(OverlayFilter.EVERY_TYPE, CRITICAL, 0.5f, false));
        assertTrue(ResultsFragment.shownOnOverlay(OverlayFilter.EVERY_TYPE, ADVISORY, 0.5f, false));
        assertTrue(ResultsFragment.shownOnOverlay(OverlayFilter.EVERY_TYPE, IGNORED, 0.5f, false));
    }

    @Test
    public void theTypeViewsHideBestMatchesUnderTheFloor() {
        for (OverlayFilter filter : new OverlayFilter[]{OverlayFilter.CRITICAL,
                OverlayFilter.CRITICAL_ADVISORY, OverlayFilter.EVERY_TYPE}) {
            assertFalse(filter.name(), ResultsFragment.shownOnOverlay(filter, CRITICAL, 0.29f, false));
            assertTrue(filter.name(), ResultsFragment.shownOnOverlay(filter, CRITICAL, 0.30f, false));
        }
    }

    @Test
    public void theLastViewShowsOnlyBestMatchesUnderTheFloor() {
        for (com.example.ml.FeatureTypeRegistry.Severity severity
                : com.example.ml.FeatureTypeRegistry.Severity.values()) {
            assertTrue(severity.name(),
                    ResultsFragment.shownOnOverlay(OverlayFilter.UNDER_FLOOR, severity, 0.29f, false));
            assertFalse(severity.name(),
                    ResultsFragment.shownOnOverlay(OverlayFilter.UNDER_FLOOR, severity, 0.30f, false));
        }
        assertFalse("an operator label is never under the floor",
                ResultsFragment.shownOnOverlay(OverlayFilter.UNDER_FLOOR, CRITICAL, 0.1f, true));
    }

    /** Every detection is in exactly one of the widest type view and the under-floor view. */
    @Test
    public void theLastTwoViewsSplitEveryDetectionBetweenThem() {
        float[] scores = {0f, 0.15f, 0.299f, 0.30f, 0.5f, 0.99f};
        for (com.example.ml.FeatureTypeRegistry.Severity severity
                : com.example.ml.FeatureTypeRegistry.Severity.values()) {
            for (float score : scores) {
                for (boolean confirmed : new boolean[]{false, true}) {
                    boolean wide = ResultsFragment.shownOnOverlay(
                            OverlayFilter.EVERY_TYPE, severity, score, confirmed);
                    boolean under = ResultsFragment.shownOnOverlay(
                            OverlayFilter.UNDER_FLOOR, severity, score, confirmed);
                    assertTrue(severity + " " + score + " " + confirmed, wide != under);
                }
            }
        }
    }

    /** A box the operator labelled is never hidden for its score: a person said what it is. */
    @Test
    public void anOperatorLabelOutranksTheScore() {
        assertTrue(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL, CRITICAL, 0.1f, true));
        assertFalse("but its type still decides",
                ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL, ADVISORY, 0.1f, true));
        assertTrue(ResultsFragment.shownOnOverlay(OverlayFilter.EVERY_TYPE, IGNORED, 0.1f, true));
    }

    @Test
    public void theButtonCyclesNarrowestToWidestThenUnderTheFloorAndBack() {
        assertEquals(OverlayFilter.CRITICAL, OverlayFilter.CRITICAL_TRUSTED.next());
        assertEquals(OverlayFilter.CRITICAL_ADVISORY, OverlayFilter.CRITICAL.next());
        assertEquals(OverlayFilter.EVERY_TYPE, OverlayFilter.CRITICAL_ADVISORY.next());
        assertEquals(OverlayFilter.UNDER_FLOOR, OverlayFilter.EVERY_TYPE.next());
        assertEquals(OverlayFilter.CRITICAL_TRUSTED, OverlayFilter.UNDER_FLOOR.next());
    }

    /** The first view: critical hardware at 0.74 or more, the calls the app does not ask about. */
    @Test
    public void theTrustedViewShowsCriticalAtTheCeilingOrAbove() {
        assertTrue(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL_TRUSTED, CRITICAL, 0.74f, false));
        assertTrue(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL_TRUSTED, CRITICAL, 0.95f, false));
        assertFalse(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL_TRUSTED, CRITICAL, 0.739f, false));
        assertFalse(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL_TRUSTED, ADVISORY, 0.95f, false));
        assertFalse(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL_TRUSTED, IGNORED, 0.95f, false));
        // An operator's label settles it, whatever the score.
        assertTrue(ResultsFragment.shownOnOverlay(OverlayFilter.CRITICAL_TRUSTED, CRITICAL, 0.40f, true));
        assertEquals("CRITICAL AT 0.74+", OverlayFilter.CRITICAL_TRUSTED.label);
    }

    @Test
    public void anUnknownStoredViewFallsBackToCriticalOnly() {
        assertEquals(OverlayFilter.CRITICAL, OverlayFilter.fromName("ALL"));
        assertEquals(OverlayFilter.CRITICAL, OverlayFilter.fromName(null));
        assertEquals(OverlayFilter.EVERY_TYPE, OverlayFilter.fromName("EVERY_TYPE"));
    }
}
