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
 * <p>Position decides, labels break close calls, and the reference's own method wins what is
 * still tied; these tests pin that order.
 */
public class AdaptivePreprocessingTest {

    private static final double MARGIN = AdaptivePreprocessing.SHORTLIST_MARGIN;

    private static Map<PreprocessingStrength, Double> scores(double medium, double strong) {
        Map<PreprocessingStrength, Double> m = new LinkedHashMap<>();
        m.put(MEDIUM, medium);
        m.put(STRONG, strong);
        return m;
    }

    @Test
    public void theMethodNearestTheOperatorsCountIsPickedAndTiesGoToTheFirst() {
        Map<String, Integer> counts = new LinkedHashMap<>();
        counts.put("a", 5);
        counts.put("b", 7);
        counts.put("c", 9);
        assertEquals("b", AdaptivePreprocessing.closestToCount(counts, 8));
        assertEquals("c", AdaptivePreprocessing.closestToCount(counts, 12));
        assertEquals("a", AdaptivePreprocessing.closestToCount(counts, 6));
        assertNull(AdaptivePreprocessing.closestToCount(counts, 0));
        assertNull(AdaptivePreprocessing.closestToCount(new LinkedHashMap<>(), 6));
    }

    @Test
    public void methodsAreSteppedThroughNearestTheCountFirst() {
        // screws_none.jpg's measured counts, abridged, in catalogue order — from before the
        // bare operators were removed, which is why two of them are here.
        Map<String, Integer> counts = new LinkedHashMap<>();
        counts.put("canny-fixed", 13);
        counts.put("canny-otsu", 5);
        counts.put("scharr", 121);
        counts.put("shipped", 6);
        counts.put("v6", 5);
        counts.put("light", 6);
        assertEquals(java.util.Arrays.asList("shipped", "light", "canny-otsu", "v6",
                        "canny-fixed", "scharr"),
                AdaptivePreprocessing.byCloseness(counts, 6));
        // The first is always the method the count picks.
        assertEquals(AdaptivePreprocessing.closestToCount(counts, 6),
                AdaptivePreprocessing.byCloseness(counts, 6).get(0));
        // Without a count there is nothing to be near: catalogue order.
        assertEquals(new java.util.ArrayList<>(counts.keySet()),
                AdaptivePreprocessing.byCloseness(counts, 0));
    }

    /** Position decides: the best fit runs when nothing is close to it. */
    @Test
    public void theBestPositionFitWins() {
        assertEquals(STRONG, AdaptivePreprocessing.choose(MEDIUM, scores(0.30, 0.05), null, MARGIN));
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(STRONG, scores(0.014, 0.20), null, MARGIN));
    }

    /** Within the margin of the best, the reference's own method keeps the job. */
    @Test
    public void theReferencesMethodWinsACloseCallWithoutLabels() {
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.10, 0.06), null, MARGIN));
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.10, 0.10), null, MARGIN));
        assertEquals(STRONG, AdaptivePreprocessing.choose(MEDIUM, scores(0.20, 0.20 - MARGIN - 1e-9), null, MARGIN));
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.20, 0.20 - MARGIN + 1e-9), null, MARGIN));
    }

    /** Between close fits, the one whose boxes carry the reference's labels wins. */
    @Test
    public void labelsDecideBetweenCloseFits() {
        Map<PreprocessingStrength, Integer> labels = new LinkedHashMap<>();
        labels.put(MEDIUM, 2);
        labels.put(STRONG, 3);
        assertEquals(STRONG, AdaptivePreprocessing.choose(MEDIUM, scores(0.06, 0.08), labels, MARGIN));
        // Equal labels: back to the reference's method.
        labels.put(STRONG, 2);
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.06, 0.08), labels, MARGIN));
    }

    /** Labels never rescue a method whose position fit is clearly worse. */
    @Test
    public void labelsDoNotOverrulePosition() {
        Map<PreprocessingStrength, Integer> labels = new LinkedHashMap<>();
        labels.put(MEDIUM, 0);
        labels.put(STRONG, 3);
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(STRONG, scores(0.05, 0.40), labels, MARGIN));
    }

    @Test
    public void theShortlistIsTheCloseFitsBestFirst() {
        Map<String, Double> s = new LinkedHashMap<>();
        s.put("a", 0.30);
        s.put("b", 0.10);
        s.put("c", Double.NaN);
        s.put("d", 0.12);
        s.put("e", 0.14);
        assertEquals(java.util.Arrays.asList("b", "d", "e"),
                AdaptivePreprocessing.shortlist(s, MARGIN, 10));
        assertEquals(java.util.Arrays.asList("b", "d"), AdaptivePreprocessing.shortlist(s, MARGIN, 2));
    }

    /** A method that failed is never chosen; with nothing scored the incumbent stands. */
    @Test
    public void aFailedMethodIsNeverChosen() {
        assertEquals(MEDIUM, AdaptivePreprocessing.choose(MEDIUM, scores(0.30, Double.NaN), null, MARGIN));
        assertEquals(STRONG, AdaptivePreprocessing.choose(MEDIUM, scores(Double.NaN, 0.30), null, MARGIN));
        Map<PreprocessingStrength, Double> withNull = scores(0.30, 0.10);
        withNull.put(LIGHT, null);
        assertEquals(STRONG, AdaptivePreprocessing.choose(LIGHT, withNull, null, MARGIN));
        assertEquals(STRONG, AdaptivePreprocessing.choose(STRONG, scores(Double.NaN, Double.NaN), null, MARGIN));
        assertEquals(LIGHT, AdaptivePreprocessing.choose(LIGHT, new LinkedHashMap<>(), null, MARGIN));
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
        for (int i = 0; i < 3; i++) layout.addSlot("screw", 0.2f + 0.2f * i, 0.5f, 0.05f, 0.05f);
        Map<String, Double> scores = new LinkedHashMap<>();
        scores.put(EdgeMethodLab.SHIPPED_ID, 0.014);
        scores.put(MEDIUM.id, 0.297);
        Map<String, Integer> labels = new LinkedHashMap<>();
        labels.put(EdgeMethodLab.SHIPPED_ID, 3);
        labels.put(MEDIUM.id, 1);
        AdaptivePreprocessing.Choice won = new AdaptivePreprocessing.Choice(
                EdgeMethodLab.SHIPPED_ID, MEDIUM.id, q, scores, labels,
                ReferenceFit.from(layout), false);
        assertTrue(won.overridden());
        assertNull("not a rung", won.rung());
        assertEquals("a layout records the method id", EdgeMethodLab.SHIPPED_ID, won.label());
        String why = won.describe();
        assertTrue(why, why.startsWith(EdgeMethodLab.nameOf(EdgeMethodLab.SHIPPED_ID)));
        assertTrue(why, why.contains("best position fit of 2 methods"));
        assertTrue(why, why.contains("3 labelled features"));
        assertTrue(why, why.contains("score 0.01"));
        assertTrue(why, why.contains("matched 3 of 3 labels"));
        assertTrue(why, why.contains("scored 0.30"));
    }

    @Test
    public void theBallotIsTheWholeCatalogueButTheAdaptiveStageItself() {
        java.util.List<String> ballot = AdaptivePreprocessing.ballot();
        assertEquals(EdgeMethodLab.catalogueSize() - 1, ballot.size());
        assertTrue(ballot.contains(EdgeMethodLab.SHIPPED_ID));
        assertTrue(ballot.contains(LIGHT.id));
        assertTrue(ballot.contains("canny-otsu"));
        assertTrue(ballot.contains("clahe-bilateral-canny"));
        assertFalse(ballot.contains(EdgeMethodLab.ADAPTIVE_ID));
        // The bare operators were removed: they buried a six-fastener plate in 12 to 121 boxes.
        for (String id : new String[]{"canny-fixed", "sobel", "scharr", "laplacian-of-gaussian",
                "morphological-gradient", "adaptive-threshold"}) {
            assertFalse(id + " was removed", ballot.contains(id));
        }
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
