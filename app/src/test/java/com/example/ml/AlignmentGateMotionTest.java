package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import android.graphics.Bitmap;
import android.graphics.Color;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

/**
 * Pins {@link AlignmentGate#measureMotion}, the "is the phone moving" estimate.
 *
 * <p>{@link AlignmentGateTest} covers the decision rule, which takes motion as a number
 * already computed. This covers where that number comes from — a frame-to-frame luminance
 * difference that decides whether a good frame counts toward the three-second hold. It gates
 * every automatic capture and had no test.
 *
 * <p>The gate is built around a null embedder: nothing on this path touches it, and doing so
 * avoids needing a live ONNX session in a JVM test. {@code evaluate()} itself, and its probe
 * short-circuit, still cannot be reached this way — that would need the embedder behind an
 * interface so a fake could stand in.
 */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = {36})
public class AlignmentGateMotionTest {

    /** A gate with no embedder — valid for everything measureMotion touches. */
    private static AlignmentGate motionOnlyGate() {
        return new AlignmentGate(null, new float[]{1f, 0f, 0f}, null);
    }

    private static Bitmap solid(int color) {
        Bitmap bmp = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[64 * 64];
        java.util.Arrays.fill(pixels, color);
        bmp.setPixels(pixels, 0, 64, 0, 0, 64, 64);
        return bmp;
    }

    @Test
    public void theFirstFrameHasNoPredecessorSoMotionIsNaN() {
        // Must be NaN, not 0: classify() reads NaN as "cannot be judged moving" and lets the
        // frame through, whereas a 0 would assert the phone was measurably still.
        float motion = motionOnlyGate().measureMotion(solid(Color.GRAY));
        assertTrue("first frame must report NaN motion", Float.isNaN(motion));
    }

    @Test
    public void twoIdenticalFramesReportNoMotion() {
        AlignmentGate gate = motionOnlyGate();
        gate.measureMotion(solid(Color.GRAY));
        assertEquals(0f, gate.measureMotion(solid(Color.GRAY)), 1e-4f);
    }

    @Test
    public void blackToWhiteIsNearlyTheFullScale() {
        // The measure is a mean absolute luminance delta on a 0-255 scale, so the largest
        // possible swing must land near 255 — this is what MAX_MOTION is calibrated against.
        AlignmentGate gate = motionOnlyGate();
        gate.measureMotion(solid(Color.BLACK));
        float motion = gate.measureMotion(solid(Color.WHITE));
        assertTrue("black to white should be near full scale, was " + motion, motion > 240f);
    }

    @Test
    public void aSmallLuminanceChangeStaysBelowTheMovingThreshold() {
        // Sensor noise on a still handheld phone runs about 1-2 on this scale. A change of
        // that order must not read as movement, or the hold could never complete.
        AlignmentGate gate = motionOnlyGate();
        gate.measureMotion(solid(Color.rgb(128, 128, 128)));
        float motion = gate.measureMotion(solid(Color.rgb(130, 130, 130)));
        assertTrue("a 2-level change should read as small, was " + motion, motion < 6f);
        assertTrue(motion > 0f);
    }

    @Test
    public void motionIsMeasuredAgainstThePreviousFrameNotTheFirst() {
        // The estimate is a rolling comparison. If it kept comparing against frame one, a
        // slow pan would look stationary once it stopped, and vice versa.
        AlignmentGate gate = motionOnlyGate();
        gate.measureMotion(solid(Color.BLACK));
        gate.measureMotion(solid(Color.WHITE));
        assertEquals("a repeated frame is stationary regardless of what came before",
                0f, gate.measureMotion(solid(Color.WHITE)), 1e-4f);
    }

    @Test
    public void resetMotionDropsThePreviousFrame() {
        // Stills call this so the previous frame's content cannot leak into a fresh score.
        AlignmentGate gate = motionOnlyGate();
        gate.measureMotion(solid(Color.BLACK));
        gate.resetMotion();
        assertTrue("after reset the next frame is a first frame again",
                Float.isNaN(gate.measureMotion(solid(Color.WHITE))));
    }
}
