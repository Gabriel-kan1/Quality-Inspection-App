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
 * Pins {@link AlignmentGate#evaluate}, in particular the scale probe's short-circuit.
 *
 * <p>The gate is documented as costing "one extra encoder pass, charged only on frames that
 * already clear the threshold". That is a performance claim about the live camera path — most
 * frames are below threshold while the operator is still hunting for the area — and nothing
 * verified it. A refactor that always ran the probe would double the per-frame encoder cost
 * silently, because every visible behaviour would stay identical.
 *
 * <p>Uses a fake {@link Encoder} returning canned vectors, so the similarity a frame scores is
 * chosen by the test rather than hoped for from a real crop.
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
    public void aFrameBelowThresholdCostsExactlyOneEncoderPass() {
        CountingEncoder encoder = new CountingEncoder(WEAK);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, REFERENCE);

        AlignmentGate.Reading r = gate.evaluate(frame());

        assertEquals("the probe must not run below the threshold", 1, encoder.calls);
        assertTrue("farGain must be NaN when the probe was skipped", Float.isNaN(r.farGain));
        assertFalse(r.inTolerance);
    }

    @Test
    public void aFrameAboveThresholdPaysForTheProbe() {
        // Frame scores 0.866, so the probe runs; the second vector is the centre crop's.
        CountingEncoder encoder = new CountingEncoder(STRONG, STRONG);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, REFERENCE);

        AlignmentGate.Reading r = gate.evaluate(frame());

        assertEquals("frame plus centre-crop probe", 2, encoder.calls);
        assertFalse("farGain must be measured once the probe runs", Float.isNaN(r.farGain));
    }

    @Test
    public void similarityIsTheCosineAgainstTheReference() {
        CountingEncoder encoder = new CountingEncoder(STRONG, STRONG);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, REFERENCE);

        assertEquals(0.866f, gate.evaluate(frame()).similarity, 1e-3f);
    }

    @Test
    public void theInputVectorIsNormalisedBeforeComparison() {
        // A non-unit vector must not inflate the similarity past 1.0: the gate normalises
        // before the dot product, so scaling the encoder's output changes nothing.
        CountingEncoder scaled = new CountingEncoder(
                new float[]{10f, 0f, 0f}, new float[]{10f, 0f, 0f});
        AlignmentGate gate = new AlignmentGate(scaled, REFERENCE, REFERENCE);

        assertEquals(1.0f, gate.evaluate(frame()).similarity, 1e-3f);
    }

    @Test
    public void aFailedEncodeIsReportedAsNoMatchRatherThanCrashing() {
        CountingEncoder failing = new CountingEncoder((float[]) null);
        AlignmentGate gate = new AlignmentGate(failing, REFERENCE, REFERENCE);

        AlignmentGate.Reading r = gate.evaluate(frame());

        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.NO_MATCH, r.fault);
        assertTrue(Float.isNaN(r.similarity));
    }

    @Test
    public void aNullFrameIsHandled() {
        CountingEncoder encoder = new CountingEncoder(STRONG);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, REFERENCE);

        AlignmentGate.Reading r = gate.evaluate(null);

        assertEquals("a null frame must not reach the encoder", 0, encoder.calls);
        assertFalse(r.inTolerance);
        assertEquals(AlignmentGate.Fault.NO_MATCH, r.fault);
    }

    @Test
    public void withNoZoomedReferenceTheTooCloseProbeIsSkipped() {
        // fromReference leaves the zoom vector null when the cropped reference fails to
        // encode; closeGain must then be NaN rather than a comparison against nothing.
        CountingEncoder encoder = new CountingEncoder(STRONG, STRONG);
        AlignmentGate gate = new AlignmentGate(encoder, REFERENCE, null);

        assertTrue(Float.isNaN(gate.evaluate(frame()).closeGain));
    }
}
