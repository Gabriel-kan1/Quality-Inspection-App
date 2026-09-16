package com.example.ui.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;

import androidx.test.core.app.ApplicationProvider;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

import java.io.File;
import java.io.FileOutputStream;

/**
 * Covers the crop cut for one side of the size comparison: the box the operator sees outlined
 * has to be the box that was measured, to the pixel, or the picture argues with the number.
 */
@RunWith(RobolectricTestRunner.class)
public class ComparisonCropsTest {

    private static Context ctx() {
        return ApplicationProvider.getApplicationContext();
    }

    @Test
    public void workingBoxRoundsTheFractionBackToDetectorPixels() {
        Rect box = ComparisonCrops.workingBox(new RectF(0.25f, 0.5f, 0.5f, 0.75f), 1280, 960);
        assertEquals(new Rect(320, 480, 640, 720), box);
    }

    /** Padding follows the box, with a floor so a tiny box still shows its surroundings. */
    @Test
    public void paddingIsProportionalWithAFloor() {
        assertEquals(Math.round(100 * ComparisonCrops.PADDING_RATIO),
                ComparisonCrops.paddingFor(new Rect(0, 0, 100, 40)));
        assertEquals(8, ComparisonCrops.paddingFor(new Rect(0, 0, 6, 6)));
    }

    /** The padded region stops at the photo's edges rather than asking for pixels outside it. */
    @Test
    public void regionIsClippedToThePhoto() {
        Rect region = ComparisonCrops.clippedRegion(new Rect(10, 10, 60, 60), 30, 1280, 960);
        assertEquals(new Rect(0, 0, 90, 90), region);
        region = ComparisonCrops.clippedRegion(new Rect(1250, 930, 1275, 955), 30, 1280, 960);
        assertEquals(new Rect(1220, 900, 1280, 960), region);
    }

    /**
     * End to end through the working-copy path on a photo smaller than the working size: the
     * crop is the padded box and the box inside it is offset by exactly the padding that
     * survived clipping.
     */
    @Test
    public void workingCopyCropCarriesTheBoxToThePixel() throws Exception {
        File photo = new File(ctx().getCacheDir(), "small.png");
        Bitmap source = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888);
        try (FileOutputStream out = new FileOutputStream(photo)) {
            source.compress(Bitmap.CompressFormat.PNG, 100, out);
        }
        source.recycle();

        // A 64x32 box at (200,100) in a 640x480 working frame — well inside, so no clipping.
        RectF fraction = new RectF(200f / 640, 100f / 480, 264f / 640, 132f / 480);
        // Uri.fromFile rather than "file://" + getAbsolutePath(): on Windows the latter yields
        // file://D:\..., where Uri reads "D:" as a host and drops the drive, so the path only
        // resolves when the JVM's working drive happens to match. This test was green on one
        // machine and null on another for exactly that reason.
        ComparisonCrops.Crop crop = ComparisonCrops.cut(ctx(), Uri.fromFile(photo).toString(),
                640, 480, fraction);
        assertNotNull(crop);
        int pad = ComparisonCrops.paddingFor(new Rect(200, 100, 264, 132));
        assertEquals(64 + 2 * pad, crop.bitmap.getWidth());
        assertEquals(32 + 2 * pad, crop.bitmap.getHeight());
        assertEquals(pad, crop.box.left, 1e-4f);
        assertEquals(pad, crop.box.top, 1e-4f);
        assertEquals(pad + 64, crop.box.right, 1e-4f);
        assertEquals(pad + 32, crop.box.bottom, 1e-4f);
        assertEquals(64, crop.boxWidthPx);
        assertEquals(32, crop.boxHeightPx);
        crop.recycle();
        assertTrue(photo.delete());
    }

    @Test
    public void anUnreadablePhotoYieldsNoCropRatherThanACrash() {
        assertNull(ComparisonCrops.cut(ctx(), "file:///nowhere/missing.jpg", 1280, 960,
                new RectF(0.4f, 0.4f, 0.5f, 0.5f)));
        assertNull(ComparisonCrops.cut(ctx(), null, 1280, 960, new RectF(0.4f, 0.4f, 0.5f, 0.5f)));
        assertNull(ComparisonCrops.cut(ctx(), "file:///x.jpg", 0, 0, new RectF(0.4f, 0.4f, 0.5f, 0.5f)));
    }
}
