package com.example.ui.view;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/**
 * The one rule that makes the side-by-side honest: the two boxes are drawn in the measured
 * ratio, with the larger pinned to the panel so the pair always fits.
 *
 * Robolectric only because the view's colour constants go through {@code Color.rgb}; the rule
 * under test is pure arithmetic.
 */
@RunWith(RobolectricTestRunner.class)
public class CropCompareViewTest {

    private static final float PANEL = 400f;

    @Test
    public void theLargerBoxIsPinnedAndTheOtherFollowsTheRatio() {
        float[] grown = CropCompareView.targetSizes(PANEL, 2f);
        assertEquals(PANEL * CropCompareView.LARGER_BOX_FRACTION, grown[1], 1e-4f);
        assertEquals(grown[1] / 2f, grown[0], 1e-4f);

        float[] shrunk = CropCompareView.targetSizes(PANEL, 0.5f);
        assertEquals(PANEL * CropCompareView.LARGER_BOX_FRACTION, shrunk[0], 1e-4f);
        assertEquals(shrunk[0] * 0.5f, shrunk[1], 1e-4f);
    }

    @Test
    public void equalSizesDrawEqualBoxes() {
        float[] same = CropCompareView.targetSizes(PANEL, 1f);
        assertEquals(same[0], same[1], 1e-4f);
        assertEquals(PANEL * CropCompareView.LARGER_BOX_FRACTION, same[0], 1e-4f);
    }

    /** An unjudged pair has no ratio to show, so neither side is drawn smaller. */
    @Test
    public void noRatioDrawsBothAtTheSameSize() {
        float[] none = CropCompareView.targetSizes(PANEL, Float.NaN);
        assertEquals(none[0], none[1], 1e-4f);
    }

    /** An extreme ratio is still shown, but the smaller box is held at a legible floor. */
    @Test
    public void aVanishinglySmallBoxIsHeldAtTheFloor() {
        float[] extreme = CropCompareView.targetSizes(PANEL, 20f);
        assertEquals(PANEL * CropCompareView.MIN_BOX_FRACTION, extreme[0], 1e-4f);
        assertEquals(PANEL * CropCompareView.LARGER_BOX_FRACTION, extreme[1], 1e-4f);
    }
}
