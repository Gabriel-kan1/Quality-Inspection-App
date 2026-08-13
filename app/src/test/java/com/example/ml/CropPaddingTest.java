package com.example.ml;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.core.Rect;
import org.robolectric.RobolectricTestRunner;

/**
 * Pins the property crop padding exists to provide: a detection should occupy roughly the
 * same share of its crop whatever its pixel size.
 *
 * Fixed padding did the opposite. At 40 px per side a 30x22 box became a 110x102 crop with
 * the fastener 6% of it, while a 201x125 box came out at 44% — so the encoder was handed a
 * different framing depending on how far away the operator stood, and a prototype averaged
 * over those crops was averaging over framing as much as over hardware.
 *
 * Runs under Robolectric only because {@link Rect} comes from the OpenCV bindings.
 */
@RunWith(RobolectricTestRunner.class)
public class CropPaddingTest {

    /** Fraction of the padded crop's area actually occupied by the detection. */
    private static double featureShare(int w, int h) {
        int pad = HardwareFeatureExtractor.cropPaddingFor(new Rect(0, 0, w, h));
        return (double) (w * h) / ((w + 2 * pad) * (h + 2 * pad));
    }

    /** The sizes that actually came out of the detector on the bundled assets. */
    private static final int[][] OBSERVED_BOXES = {
            {30, 22}, {66, 89}, {84, 117}, {93, 87}, {129, 111}, {201, 125}
    };

    @Test
    public void framingIsConsistentAcrossTheObservedSizeRange() {
        double min = Double.MAX_VALUE, max = 0;
        for (int[] box : OBSERVED_BOXES) {
            double share = featureShare(box[0], box[1]);
            min = Math.min(min, share);
            max = Math.max(max, share);
        }
        // Fixed padding spanned 6%-44%, a 7x swing. Proportional padding must stay tight.
        assertTrue("smallest feature must still fill most of its crop, was " + min, min > 0.4);
        assertTrue("framing must not vary much with size, spread was " + (max - min),
                max - min < 0.15);
    }

    @Test
    public void paddingScalesWithTheDetection() {
        int small = HardwareFeatureExtractor.cropPaddingFor(new Rect(0, 0, 30, 22));
        int large = HardwareFeatureExtractor.cropPaddingFor(new Rect(0, 0, 201, 125));
        assertTrue("a larger feature gets proportionally more context", large > small);
    }

    /**
     * Both axes get the same padding. Padding each side by its own dimension would stretch an
     * elongated feature's context along the direction it is already long.
     */
    @Test
    public void paddingIsSquareRegardlessOfAspect() {
        assertEquals(HardwareFeatureExtractor.cropPaddingFor(new Rect(0, 0, 120, 20)),
                HardwareFeatureExtractor.cropPaddingFor(new Rect(0, 0, 20, 120)));
    }

    /** A tiny detection must still get a pixel or two, since its contour edge is uncertain. */
    @Test
    public void aTinyDetectionStillGetsSomeContext() {
        assertTrue(HardwareFeatureExtractor.cropPaddingFor(new Rect(0, 0, 4, 3)) >= 2);
    }
}
