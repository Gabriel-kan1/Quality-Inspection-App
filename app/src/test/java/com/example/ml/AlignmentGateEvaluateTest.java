package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import android.graphics.Bitmap;
import android.graphics.Color;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.util.ArrayList;
import java.util.List;

/**
 * Pins {@link AlignmentGate#evaluate}: what it costs per frame, and what it does when the
 * framing search cannot run.
 *
 * <p>OpenCV's native libraries are not loaded in a JVM test, so {@link TemplateMatcher}
 * returns nothing here. That is not a gap — it is the same state a device with a missing or
 * ABI-mismatched {@code .so} is in, and the gate's behaviour there is a safety property
 * worth pinning: it must refuse rather than pass.
 *
 * <p>Uses a fake {@link Encoder} returning canned vectors, so the similarity a frame scores
 * is chosen by the test rather than hoped for from a real crop.
 */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = {36})
public class AlignmentGateEvaluateTest {

    /** Records every call so the number of encoder passes can be asserted. */
    private static final class CountingEncoder implements Encoder {
        final List<float[]> queued = new ArrayList<>();
        int calls;

        CountingEncoder(float[]... vectors) {
            for (float[] v : vectors) queued.add(v);
        }

        @Override
        public float[] embed(Bitmap crop) {
            float[] v = queued.get(Math.min(calls, queued.size() - 1));
            calls++;
            return v == null ? null : v.clone();
        }
    }

    /** Unit vectors in a 2-D slice, so cosine similarity is cos(angle) and easy to choose. */
    private static float[] at(double degrees) {
        double r = Math.toRadians(degrees);
        return new float[]{(float) Math.cos(r), (float) Math.sin(r), 0f};
    }

    private static final float[] REFERENCE = at(0);       // similarity 1.0 against itself
    private static final float[] STRONG = at(30);         // cos 30 = 0.866, clears 0.70
    private static final float[] WEAK = at(60);           // cos 60 = 0.500, below 0.70

    private static Bitmap frame() {
        Bitmap bmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
        int[] px = new int[64 * 64];
        java.util.Arrays.fill(px, Color.GRAY);
        bmp.setPixels(px, 0, 64, 0, 0, 64, 64);
        return bmp;
    }

    @Test
    public void everyFrameCostsExactlyOneEncoderPass() {
        // The framing search runs on gradients, not embeddings, so rebuilding the gate around
        // it must not have added encoder work per frame. It previously cost two passes on
        // every frame that cleared the threshold.
        CountingEncoder encoder = new CountingEncoder(STRONG);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, null);

        gate.evaluate(frame(), 0L);

        assertEquals("one pass for the area check and nothing more", 1, encoder.calls);
    }

    @Test
    public void aFrameBelowTheAreaThresholdIsRejectedWithoutSearchingFraming() {
        CountingEncoder encoder = new CountingEncoder(WEAK);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, null);

        AlignmentGate.Reading r = gate.evaluate(frame(), 0L);

        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.WEAK_MATCH, r.fault);
        assertTrue("no framing was measured, so it must not be reported",
                Float.isNaN(r.scale));
    }

    @Test
    public void similarityIsTheCosineAgainstTheReference() {
        CountingEncoder encoder = new CountingEncoder(STRONG);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, null);

        assertEquals(0.866f, gate.evaluate(frame(), 0L).similarity, 1e-3f);
    }

    @Test
    public void theInputVectorIsNormalisedBeforeComparison() {
        // A non-unit vector must not inflate the similarity past 1.0.
        CountingEncoder scaled = new CountingEncoder(new float[]{10f, 0f, 0f});
        AlignmentGate gate = new AlignmentGate(scaled, REFERENCE, null);

        assertEquals(1.0f, gate.evaluate(frame(), 0L).similarity, 1e-3f);
    }

    /**
     * The area matches but nothing can measure the framing, so the gate must refuse.
     *
     * <p>This is the state a device without working OpenCV natives is permanently in. The
     * old gate would have passed such a frame on similarity alone, which is precisely the
     * misframed capture the rebuild exists to stop.
     */
    @Test
    public void anAreaMatchWithoutAFramingSearchIsRefused() {
        CountingEncoder encoder = new CountingEncoder(STRONG);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, null);

        AlignmentGate.Reading r = gate.evaluate(frame(), 0L);

        assertFalse("similarity alone must never satisfy the gate", r.inTolerance);
        assertEquals(AlignmentGate.Fault.NO_FRAMING, r.fault);
    }

    @Test
    public void aFailedEncodeIsReportedAsNoMatchRatherThanCrashing() {
        CountingEncoder failing = new CountingEncoder((float[]) null);
        AlignmentGate gate = new AlignmentGate(failing, REFERENCE, null);

        AlignmentGate.Reading r = gate.evaluate(frame(), 0L);

        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.NO_MATCH, r.fault);
        assertTrue(Float.isNaN(r.similarity));
    }

    @Test
    public void aNullFrameIsHandled() {
        CountingEncoder encoder = new CountingEncoder(STRONG);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, null);

        AlignmentGate.Reading r = gate.evaluate(null, 0L);

        assertEquals("a null frame must not reach the encoder", 0, encoder.calls);
        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.NO_MATCH, r.fault);
    }
}
