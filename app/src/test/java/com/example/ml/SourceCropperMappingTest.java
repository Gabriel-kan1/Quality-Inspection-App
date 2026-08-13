package com.example.ml;

import static org.junit.Assert.assertEquals;

import android.graphics.Rect;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/**
 * Covers the upright-to-raw rectangle mapping used when cropping from the source photo.
 *
 * The working image has EXIF rotation applied; {@code BitmapRegionDecoder} does not, and
 * reads raw sensor coordinates. An error here would not soften a crop, it would cut a
 * completely different part of the picture and hand it to the classifier as a fastener — so
 * the mapping is checked against the corners it has to preserve.
 */
@RunWith(RobolectricTestRunner.class)
public class SourceCropperMappingTest {

    /** A portrait sensor image: raw 400 wide, 300 tall. */
    private static final int RAW_W = 400;
    private static final int RAW_H = 300;

    @Test
    public void noRotationIsIdentity() {
        Rect upright = new Rect(10, 20, 60, 90);
        assertEquals(upright, SourceCropper.toRawRect(upright, RAW_W, RAW_H, 0));
    }

    /**
     * At 90 the upright frame is raw with its axes swapped, so a box hard against the upright
     * top-left must come back against the raw bottom-left.
     */
    @Test
    public void quarterTurnMapsTheUprightOriginToTheRawBottomLeft() {
        // Upright frame is 300 wide (RAW_H) by 400 tall (RAW_W).
        Rect corner = new Rect(0, 0, 30, 40);
        Rect raw = SourceCropper.toRawRect(corner, RAW_W, RAW_H, 90);
        assertEquals(new Rect(0, RAW_H - 30, 40, RAW_H), raw);
    }

    @Test
    public void halfTurnMirrorsBothAxes() {
        Rect corner = new Rect(0, 0, 30, 40);
        Rect raw = SourceCropper.toRawRect(corner, RAW_W, RAW_H, 180);
        assertEquals(new Rect(RAW_W - 30, RAW_H - 40, RAW_W, RAW_H), raw);
    }

    @Test
    public void threeQuarterTurnMapsTheUprightOriginToTheRawTopRight() {
        Rect corner = new Rect(0, 0, 30, 40);
        Rect raw = SourceCropper.toRawRect(corner, RAW_W, RAW_H, 270);
        assertEquals(new Rect(RAW_W - 40, 0, RAW_W, 30), raw);
    }

    /** Every rotation must preserve the area being asked for; only its placement moves. */
    @Test
    public void areaIsPreservedUnderEveryRotation() {
        Rect upright = new Rect(12, 34, 78, 90);
        int expected = upright.width() * upright.height();
        for (int rotation : new int[]{0, 90, 180, 270}) {
            Rect raw = SourceCropper.toRawRect(upright, RAW_W, RAW_H, rotation);
            assertEquals("rotation " + rotation, expected, raw.width() * raw.height());
        }
    }

    /** Mapping forward then back through the opposite rotation must return the original. */
    @Test
    public void quarterTurnsAreInverses() {
        Rect upright = new Rect(12, 34, 78, 90);
        // Upright space at 90 is RAW_H x RAW_W, so the inverse runs in those dimensions.
        Rect raw = SourceCropper.toRawRect(upright, RAW_W, RAW_H, 90);
        Rect back = SourceCropper.toRawRect(raw, RAW_H, RAW_W, 270);
        assertEquals(upright, back);
    }

    @Test
    public void outOfRangeAnglesAreNormalised() {
        Rect upright = new Rect(10, 20, 60, 90);
        assertEquals(SourceCropper.toRawRect(upright, RAW_W, RAW_H, 90),
                SourceCropper.toRawRect(upright, RAW_W, RAW_H, 450));
        assertEquals(SourceCropper.toRawRect(upright, RAW_W, RAW_H, 270),
                SourceCropper.toRawRect(upright, RAW_W, RAW_H, -90));
    }
}
