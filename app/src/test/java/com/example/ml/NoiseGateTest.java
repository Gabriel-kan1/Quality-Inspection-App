package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import android.graphics.RectF;

import com.example.data.entity.ClassificationEvent;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Pins what the noise filter may and may not do.
 *
 * The filter removes dialogs, and the one thing that must never be removed is a dialog about
 * a fastener. So most of these are about refusal: before the evidence exists, when the
 * evidence contains a critical, when a critical type is a near-tie, when the crop resembles
 * nothing at all, and when the crop is ambiguous rather than merely rejected. The single
 * positive case is the one the feature exists for.
 *
 * Robolectric only for ReferenceLibrary's android.graphics.Color static block.
 */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = {36})
public class NoiseGateTest {

    /** A catalogue with one ignored type, one advisory, and the critical hardware. */
    private static final NoiseGate.SeverityLookup SEVERITY = type -> {
        if ("surface".equals(type)) return FeatureTypeRegistry.Severity.IGNORED;
        if ("scratches".equals(type)) return FeatureTypeRegistry.Severity.ADVISORY;
        return FeatureTypeRegistry.Severity.CRITICAL;
    };

    private static ReferenceLibrary library() {
        ReferenceLibrary lib = new ReferenceLibrary();
        lib.setThresholds(0.74f, 0.02f);
        return lib;
    }

    /** A reviewed below-threshold event whose nearest prototype was {@code closest}. */
    private static ClassificationEvent reviewed(String closest, float score, String truth) {
        ClassificationEvent e = new ClassificationEvent();
        e.setPartId("P");
        e.setEncoderId(FeatureEmbedder.ENCODER_ID);
        e.setPredictedType(closest);
        e.setPredictedScore(score);
        e.setReason(ReferenceLibrary.Reason.BELOW_THRESHOLD.name());
        e.setConfirmedType(truth);
        e.setReviewed(true);
        return e;
    }

    private static List<ClassificationEvent> noiseHistory(int clean, int critical) {
        List<ClassificationEvent> out = new ArrayList<>();
        for (int i = 0; i < clean; i++) out.add(reviewed("surface", 0.66f, "surface"));
        for (int i = 0; i < critical; i++) out.add(reviewed("surface", 0.66f, "stud"));
        return out;
    }

    /** A live classification: rejected below threshold, nearest {@code closest}. */
    private static ReferenceLibrary.Classification live(String closest, float score,
                                                        ReferenceLibrary.Reason reason,
                                                        Object... typeScorePairs) {
        Map<String, Float> scores = new LinkedHashMap<>();
        scores.put(closest, score);
        for (int i = 0; i + 1 < typeScorePairs.length; i += 2) {
            scores.put((String) typeScorePairs[i], (Float) typeScorePairs[i + 1]);
        }
        String runnerUp = typeScorePairs.length > 0 ? (String) typeScorePairs[0] : null;
        float runnerUpScore = typeScorePairs.length > 0 ? (Float) typeScorePairs[1] : -1f;
        boolean unknown = reason != ReferenceLibrary.Reason.ACCEPTED;
        return new ReferenceLibrary.Classification(
                unknown ? ReferenceLibrary.UNKNOWN : closest, closest, score, "low",
                runnerUp, runnerUpScore, reason, scores);
    }

    private static ReferenceLibrary.Classification noiseLooking() {
        return live("surface", 0.66f, ReferenceLibrary.Reason.BELOW_THRESHOLD,
                "scratches", 0.55f, "stud", 0.40f);
    }

    @Test
    public void nothingIsSuppressedBeforeTheEvidenceExists() {
        NoiseGate gate = NoiseGate.fit(noiseHistory(NoiseGate.MIN_EVIDENCE - 1, 0), SEVERITY, library());
        assertFalse(gate.evidence.mature);
        assertFalse("nineteen clean reviews is not yet a conclusion",
                gate.suppresses(noiseLooking()));
    }

    /** The case the filter exists for. */
    @Test
    public void aMatureGateSuppressesACropThatResemblesAnIgnoredType() {
        NoiseGate gate = NoiseGate.fit(noiseHistory(NoiseGate.MIN_EVIDENCE, 0), SEVERITY, library());
        assertTrue(gate.evidence.mature);
        assertTrue(gate.suppresses(noiseLooking()));
    }

    /**
     * One fastener among twenty noise-looking crops is enough to keep asking. The rate limit
     * is 2%, which at the minimum sample rounds to zero — deliberately.
     */
    @Test
    public void oneCriticalInTwentyKeepsTheGateShut() {
        NoiseGate gate = NoiseGate.fit(noiseHistory(19, 1), SEVERITY, library());
        assertFalse(gate.evidence.mature);
        assertFalse(gate.suppresses(noiseLooking()));
        assertEquals(1, gate.evidence.critical);
    }

    /** The allowance relaxes only as the sample grows enough for one slip to be plausible. */
    @Test
    public void theCriticalAllowanceScalesWithTheSample() {
        assertTrue("1 in 50 is inside the 2% allowance",
                NoiseGate.fit(noiseHistory(49, 1), SEVERITY, library()).evidence.mature);
        assertFalse("2 in 50 is not",
                NoiseGate.fit(noiseHistory(48, 2), SEVERITY, library()).evidence.mature);
    }

    /**
     * The condition the evidence cannot replay, enforced live: a crop that is nearly as much
     * "stud" as "surface" is a question, however mature the gate.
     */
    @Test
    public void aNearTieWithACriticalTypeIsNeverSuppressed() {
        NoiseGate gate = NoiseGate.fit(noiseHistory(40, 0), SEVERITY, library());
        assertTrue(gate.evidence.mature);

        ReferenceLibrary.Classification tie = live("surface", 0.66f,
                ReferenceLibrary.Reason.BELOW_THRESHOLD, "stud", 0.65f);
        assertFalse("stud within the margin must reach the operator", gate.suppresses(tie));

        ReferenceLibrary.Classification clear = live("surface", 0.66f,
                ReferenceLibrary.Reason.BELOW_THRESHOLD, "stud", 0.60f);
        assertTrue("stud well below the margin does not", gate.suppresses(clear));
    }

    /** A near-tie between two non-critical types is still noise. */
    @Test
    public void aNearTieBetweenIgnoredAndAdvisoryIsStillNoise() {
        NoiseGate gate = NoiseGate.fit(noiseHistory(40, 0), SEVERITY, library());
        ReferenceLibrary.Classification c = live("surface", 0.66f,
                ReferenceLibrary.Reason.BELOW_THRESHOLD, "scratches", 0.655f, "stud", 0.30f);
        assertTrue(gate.suppresses(c));
    }

    /** Ambiguous means it cleared the threshold and two types want it. That is a question. */
    @Test
    public void anAmbiguousCropIsAQuestionNotNoise() {
        NoiseGate gate = NoiseGate.fit(noiseHistory(40, 0), SEVERITY, library());
        ReferenceLibrary.Classification c = live("surface", 0.80f,
                ReferenceLibrary.Reason.AMBIGUOUS, "scratches", 0.79f);
        assertFalse(gate.suppresses(c));
    }

    /** Below the noise floor it resembles nothing, which may be a novel fastener. */
    @Test
    public void aCropResemblingNothingIsNotNoise() {
        NoiseGate gate = NoiseGate.fit(noiseHistory(40, 0), SEVERITY, library());
        ReferenceLibrary.Classification c = live("surface", ActiveLearningPolicy.NOISE_FLOOR - 0.01f,
                ReferenceLibrary.Reason.BELOW_THRESHOLD, "stud", 0.20f);
        assertFalse(gate.suppresses(c));
    }

    /** Only crops nearest an ignored type are candidates — never advisory, never critical. */
    @Test
    public void onlyCropsNearestAnIgnoredTypeAreCandidates() {
        NoiseGate gate = NoiseGate.fit(noiseHistory(40, 0), SEVERITY, library());
        assertFalse("nearest an advisory type", gate.suppresses(
                live("scratches", 0.66f, ReferenceLibrary.Reason.BELOW_THRESHOLD, "surface", 0.50f)));
        assertFalse("nearest a critical type", gate.suppresses(
                live("stud", 0.66f, ReferenceLibrary.Reason.BELOW_THRESHOLD, "surface", 0.50f)));
    }

    /** Skips carry no information about what the crop was. */
    @Test
    public void skippedReviewsAreNotEvidence() {
        List<ClassificationEvent> history = new ArrayList<>();
        for (int i = 0; i < 40; i++) history.add(reviewed("surface", 0.66f, ReferenceLibrary.UNKNOWN));
        NoiseGate gate = NoiseGate.fit(history, SEVERITY, library());
        assertEquals(0, gate.evidence.sampleSize);
        assertFalse(gate.evidence.mature);
    }

    /** A score from a retired encoder says nothing about resemblance under this one. */
    @Test
    public void eventsFromARetiredEncoderAreIgnored() {
        List<ClassificationEvent> history = noiseHistory(40, 0);
        for (ClassificationEvent e : history) e.setEncoderId("some-older-encoder");
        NoiseGate gate = NoiseGate.fit(history, SEVERITY, library());
        assertEquals(0, gate.evidence.sampleSize);
    }

    /** Evidence is only counted over the noise population, not over every review. */
    @Test
    public void reviewsOutsideTheNoisePopulationDoNotCount() {
        List<ClassificationEvent> history = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            ClassificationEvent e = reviewed("stud", 0.90f, "stud");
            e.setReason(ReferenceLibrary.Reason.ACCEPTED.name());
            history.add(e);
        }
        assertEquals(0, NoiseGate.fit(history, SEVERITY, library()).evidence.sampleSize);
    }

    /** Re-grading the ignored type to critical must retroactively close the gate. */
    @Test
    public void reGradingTheIgnoredTypeAsCriticalClosesTheGate() {
        List<ClassificationEvent> history = noiseHistory(40, 0);
        NoiseGate.SeverityLookup everythingCritical = type -> FeatureTypeRegistry.Severity.CRITICAL;
        NoiseGate gate = NoiseGate.fit(history, everythingCritical, library());
        assertEquals("nothing is in the noise population any more", 0, gate.evidence.sampleSize);
        assertFalse(gate.suppresses(noiseLooking()));
    }

    @Test
    public void applyRetypesTheFeatureInPlaceAndMarksItAsAutomatic() {
        NoiseGate gate = NoiseGate.fit(noiseHistory(40, 0), SEVERITY, library());
        HardwareFeatureExtractor.ExtractedFeature noise = new HardwareFeatureExtractor.ExtractedFeature(
                ReferenceLibrary.UNKNOWN, new RectF(0.1f, 0.1f, 0.2f, 0.2f), 0,
                "3:? (surface 0.66)", 0.66f, null, noiseLooking());
        HardwareFeatureExtractor.ExtractedFeature fastener = new HardwareFeatureExtractor.ExtractedFeature(
                "stud", new RectF(0.5f, 0.5f, 0.6f, 0.6f), 0, "4:stud 0.91", 0.91f, null,
                live("stud", 0.91f, ReferenceLibrary.Reason.ACCEPTED, "screw", 0.70f));

        List<HardwareFeatureExtractor.ExtractedFeature> features = new ArrayList<>();
        features.add(noise);
        features.add(fastener);
        assertEquals(1, gate.apply(features));

        assertEquals("surface", noise.type);
        assertEquals("3:surface 0.66 (auto)", noise.label);
        assertEquals(ReferenceLibrary.Reason.AUTO_IGNORED, noise.classification.reason);
        assertEquals("the score is carried, not inflated", 0.66f, noise.classification.score, 1e-6f);

        assertEquals("an accepted fastener is untouched", "stud", fastener.type);
        assertEquals(ReferenceLibrary.Reason.ACCEPTED, fastener.classification.reason);
    }

    @Test
    public void anInactiveGateNeverFires() {
        NoiseGate gate = NoiseGate.inactive(SEVERITY);
        assertFalse(gate.evidence.mature);
        assertFalse(gate.suppresses(noiseLooking()));
    }
}
