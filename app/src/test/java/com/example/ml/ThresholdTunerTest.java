package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.example.data.entity.ClassificationEvent;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.util.ArrayList;
import java.util.List;

/**
 * Pins the guards ThresholdTuner's javadoc promises.
 *
 * These matter more than most tests here: the tuner rewrites the decision boundary the
 * classifier runs on, from data the operator supplied under a UI that preselects the
 * classifier's own guess. Every guard below is the difference between "learns from
 * corrections" and "drifts wherever a run of easy confirmations pushes it", and none of
 * them was covered.
 *
 * Robolectric is needed only because the tuner logs its decisions through android.util.Log,
 * which throws under a plain JVM test rather than returning a default.
 */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = {36})
public class ThresholdTunerTest {

    private static ClassificationEvent event(String predicted, float score,
                                             String runnerUp, float runnerUpScore,
                                             String confirmed) {
        return event(predicted, score, runnerUp, runnerUpScore, confirmed,
                FeatureEmbedder.ENCODER_ID);
    }

    private static ClassificationEvent event(String predicted, float score,
                                             String runnerUp, float runnerUpScore,
                                             String confirmed, String encoderId) {
        ClassificationEvent e = new ClassificationEvent();
        e.setPredictedType(predicted);
        e.setPredictedScore(score);
        e.setRunnerUpType(runnerUp);
        e.setRunnerUpScore(runnerUpScore);
        e.setConfirmedType(confirmed);
        e.setEncoderId(encoderId);
        e.setReviewed(true);
        return e;
    }

    /**
     * A history the defaults handle badly: every crop is genuinely a stud and scored as
     * one, but at 0.60-0.70 — under DEFAULT_SIMILARITY_THRESHOLD (0.74), so the shipping
     * config calls them all unknown. A lower threshold recovers every one.
     */
    private static List<ClassificationEvent> historyFavouringALowerThreshold(int n) {
        List<ClassificationEvent> events = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            float score = 0.60f + (i % 11) * 0.01f;
            events.add(event("stud", score, "bolt_head", score - 0.20f, "stud"));
        }
        return events;
    }

    @Test
    public void refusesToFitBelowMinEvents() {
        assertNull("a handful of labels is noise, not evidence",
                ThresholdTuner.fit(historyFavouringALowerThreshold(ThresholdTuner.MIN_EVENTS - 1)));
    }

    @Test
    public void fitsOnceEnoughLabelsExist() {
        ThresholdTuner.Tuned tuned =
                ThresholdTuner.fit(historyFavouringALowerThreshold(ThresholdTuner.MIN_EVENTS));

        assertNotNull("this history is exactly what tuning is for", tuned);
        assertTrue("the fit must drop below the default to accept these scores",
                tuned.similarityThreshold < ReferenceLibrary.DEFAULT_SIMILARITY_THRESHOLD);
        assertTrue("and must beat the baseline it replaced",
                tuned.accuracy >= tuned.baselineAccuracy + ThresholdTuner.MIN_IMPROVEMENT);
        assertEquals(ThresholdTuner.MIN_EVENTS, tuned.sampleSize);
    }

    @Test
    public void neverDisablesTheAmbiguityCheck() {
        // Every event here is a runaway winner, so the search has no reason to keep any
        // margin at all — which is precisely when it would switch the check off.
        List<ClassificationEvent> events = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            events.add(event("stud", 0.70f, "bolt_head", 0.10f, "stud"));
        }
        ThresholdTuner.Tuned tuned = ThresholdTuner.fit(events);

        assertNotNull(tuned);
        assertTrue("a zero margin means committing on coin-flips",
                tuned.marginThreshold >= 0.02f);
    }

    @Test
    public void refusesToFitAnInconsistentHistory() {
        // The same score confirmed as three different types: no threshold pair can be right
        // more than about a third of the time, which is the MIN_ACCURACY floor's whole job.
        List<ClassificationEvent> events = new ArrayList<>();
        String[] truths = {"stud", "bolt_head", "screw"};
        for (int i = 0; i < 45; i++) {
            events.add(event("stud", 0.80f, "bolt_head", 0.50f, truths[i % 3]));
        }

        assertNull("beating a bad baseline is not evidence of a good threshold",
                ThresholdTuner.fit(events));
    }

    @Test
    public void ignoresEventsFromAnotherEncoder() {
        List<ClassificationEvent> mixed = new ArrayList<>();
        for (ClassificationEvent e : historyFavouringALowerThreshold(ThresholdTuner.MIN_EVENTS)) {
            mixed.add(e);
        }
        for (int i = 0; i < 50; i++) {
            mixed.add(event("stud", 0.95f, "bolt_head", 0.90f, "stud", "retired-encoder-512"));
        }

        assertEquals("only same-encoder events may be counted",
                ThresholdTuner.MIN_EVENTS, ThresholdTuner.forCurrentEncoder(mixed).size());

        ThresholdTuner.Tuned tuned = ThresholdTuner.fit(mixed);
        assertNotNull(tuned);
        assertEquals("the foreign events must not inflate the sample",
                ThresholdTuner.MIN_EVENTS, tuned.sampleSize);
    }

    @Test
    public void aHistoryOfOnlyForeignEventsCannotBeFitted() {
        List<ClassificationEvent> foreign = new ArrayList<>();
        for (int i = 0; i < 60; i++) {
            foreign.add(event("stud", 0.95f, "bolt_head", 0.50f, "stud", "retired-encoder-512"));
        }
        assertNull(ThresholdTuner.fit(foreign));
    }

    @Test
    public void keepsTheDefaultsWhenTheyAlreadyWin() {
        // Scores that the shipping defaults already classify correctly: there is nothing
        // to gain, so the tuner must leave well alone rather than drift.
        List<ClassificationEvent> events = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            events.add(event("stud", 0.90f, "bolt_head", 0.40f, "stud"));
        }
        assertNull("no improvement available means no change",
                ThresholdTuner.fit(events));
    }

    @Test
    public void anEmptyOrNullHistoryIsHandled() {
        assertNull(ThresholdTuner.fit(null));
        assertNull(ThresholdTuner.fit(new ArrayList<>()));
        assertTrue(ThresholdTuner.forCurrentEncoder(null).isEmpty());
    }
}
