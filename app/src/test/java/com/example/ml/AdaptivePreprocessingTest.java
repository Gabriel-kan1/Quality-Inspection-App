package com.example.ml;

import static com.example.ml.PreprocessingStrength.LIGHT;
import static com.example.ml.PreprocessingStrength.MEDIUM;
import static com.example.ml.PreprocessingStrength.STRONG;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Covers the decision the adaptive edge stage makes once every rung has been scored — the
 * part that can be tested without OpenCV.
 *
 * <p>The margin is the load-bearing constant. {@code python/adaptive_preprocessing_study.py}
 * found that at 0.03 the size test made bad overrides (a rung winning on size while
 * over-detecting everywhere else) and that at 0.08 it never chose worse than the image-quality
 * rung; these tests pin that behaviour rather than the number alone.
 */
public class AdaptivePreprocessingTest {

    private static final double MARGIN = AdaptivePreprocessing.AGREEMENT_MARGIN;

    private static Map<PreprocessingStrength, Double> scores(double medium, double strong) {
        Map<PreprocessingStrength, Double> m = new LinkedHashMap<>();
        m.put(MEDIUM, medium);
        m.put(STRONG, strong);
        return m;
    }

    @Test
    public void theSuggestedRungStandsWhenNothingBeatsItByTheMargin() {
        // Strong is better, but by less than size jitter between two photographs of the same
        // hardware (median 0.058): not evidence.
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.10, 0.06), MARGIN));
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.10, 0.10), MARGIN));
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.10, 0.30), MARGIN));
    }

    @Test
    public void aRungThatAgreesClearlyBetterDisplacesIt() {
        // The screws2 -> screws_two pair under noise: strong suggested, medium at 0.014
        // against strong's 0.097. Medium found all four fasteners at the reference's sizes.
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(STRONG, scores(0.014, 0.097), MARGIN));
        assertEquals(STRONG, AdaptivePreprocessing.choose(MEDIUM, scores(0.30, 0.05), MARGIN));
    }

    @Test
    public void theMarginIsAppliedExactly() {
        assertEquals(STRONG, AdaptivePreprocessing.choose(MEDIUM, scores(0.20, 0.20 - MARGIN - 1e-9), MARGIN));
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.20, 0.20 - MARGIN + 1e-9), MARGIN));
    }

    /**
     * A rung whose boxes could not be registered onto the part found something other than
     * the part. It is never chosen, and if it was the suggestion the best aligned rung wins
     * outright — there is nothing for the margin to protect.
     */
    @Test
    public void anUnalignedRungIsNeverChosenAndAnUnalignedSuggestionYields() {
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.30, Double.NaN), MARGIN));
        assertEquals(STRONG, AdaptivePreprocessing.choose(MEDIUM, scores(Double.NaN, 0.30), MARGIN));

        Map<PreprocessingStrength, Double> withNull = scores(0.30, 0.10);
        withNull.put(LIGHT, null);
        assertEquals(STRONG, AdaptivePreprocessing.choose(LIGHT, withNull, MARGIN));
    }

    @Test
    public void withNothingAlignedTheSuggestionStands() {
        assertEquals(STRONG, AdaptivePreprocessing.choose(STRONG, scores(Double.NaN, Double.NaN), MARGIN));
        assertEquals(LIGHT, AdaptivePreprocessing.choose(LIGHT, new LinkedHashMap<>(), MARGIN));
    }

    /** A suggestion the candidate set did not include is still the fallback, never a crash. */
    @Test
    public void aSuggestionAbsentFromTheScoresFallsBackToTheBestAligned() {
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(LIGHT, scores(0.10, 0.12), MARGIN));
    }

    @Test
    public void theChoiceDescribesItselfForTheBanner() {
        ImageQuality q = new ImageQuality(2.7, 62);
        AdaptivePreprocessing.Choice start = new AdaptivePreprocessing.Choice(
                MEDIUM.id, MEDIUM.id, q, new LinkedHashMap<>(), null, false);
        assertFalse(start.overridden());
        assertEquals("medium", start.label());
        String text = start.describe();
        assertTrue(text, text.startsWith("Hybrid v6"));
        assertTrue(text, text.contains("no labelled reference"));

        FeatureLayout layout = new FeatureLayout();
        layout.setImageSize(1280, 960);
        for (int i = 0; i < 4; i++) layout.addSlot("screw", 0.2f + 0.2f * i, 0.5f, 0.05f, 0.05f);
        Map<String, Double> scores = new LinkedHashMap<>();
        scores.put(EdgeMethodLab.SHIPPED_ID, 0.014);
        scores.put(MEDIUM.id, 0.297);
        AdaptivePreprocessing.Choice won = new AdaptivePreprocessing.Choice(
                EdgeMethodLab.SHIPPED_ID, MEDIUM.id, q, scores, SizePrior.from(layout), false);
        assertTrue(won.overridden());
        assertNull("not a rung", won.rung());
        assertEquals("a layout records the method id", EdgeMethodLab.SHIPPED_ID, won.label());
        String why = won.describe();
        assertTrue(why, why.startsWith(EdgeMethodLab.nameOf(EdgeMethodLab.SHIPPED_ID)));
        assertTrue(why, why.contains("best of 2 methods"));
        assertTrue(why, why.contains("score 0.01"));
        assertTrue(why, why.contains("scored 0.30"));
        assertTrue(why, why.contains("4 slots sized"));
    }

    @Test
    public void theBallotIsTheWholeCatalogueButTheAdaptiveStageItself() {
        java.util.List<String> ballot = AdaptivePreprocessing.ballot();
        assertEquals(EdgeMethodLab.catalogueSize() - 1, ballot.size());
        assertTrue(ballot.contains(EdgeMethodLab.SHIPPED_ID));
        assertTrue(ballot.contains(LIGHT.id));
        assertFalse(ballot.contains(EdgeMethodLab.ADAPTIVE_ID));
    }

    @Test
    public void theReferencesOwnMethodIsReadFromEveryLayoutVintage() {
        FeatureLayout l = new FeatureLayout();
        assertEquals("predates the choice: the shipped stack", EdgeMethodLab.SHIPPED_ID,
                l.referenceMethodId());
        l.setEdgeMethodId(EdgeMethodLab.ADAPTIVE_ID);
        assertNull("adaptive with no record of what ran", l.referenceMethodId());
        l.setPreprocessingRung("light");
        assertEquals("a rung label", LIGHT.id, l.referenceMethodId());
        l.setPreprocessingRung(EdgeMethodLab.SHIPPED_ID);
        assertEquals("a method id", EdgeMethodLab.SHIPPED_ID, l.referenceMethodId());
        l.setPreprocessingRung(null);
        l.setEdgeMethodId("hybrid-v4");
        assertEquals("a fixed method", "hybrid-v4", l.referenceMethodId());
    }
}
