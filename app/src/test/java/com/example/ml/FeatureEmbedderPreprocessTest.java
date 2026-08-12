package com.example.ml;

import static org.junit.Assert.assertEquals;

import android.graphics.Bitmap;
import android.graphics.Color;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;
import org.robolectric.annotation.Config;

import java.nio.FloatBuffer;

/**
 * Pins {@link FeatureEmbedder#preprocess}, the step that turns a crop into the tensor the
 * encoder sees.
 *
 * <p>It had no test, and it is the one piece of the pipeline that is re-implemented outside
 * Java: {@code python/pair_ground_truth.py} and {@code python/scale_probe_study.py} both
 * carry their own copy so their similarities match the app's. Every threshold in
 * {@link AlignmentGate} and {@link ReferenceLibrary} was fitted through those copies, so a
 * silent change here — a channel order, a normalisation constant — would invalidate the
 * constants without breaking anything visibly.
 *
 * <p>Robolectric supplies the Bitmap implementation. The assertions deliberately stick to
 * tensor shape and per-channel normalisation on a uniform image, which do not depend on how
 * faithfully the shadow resamples pixels.
 */
@RunWith(RobolectricTestRunner.class)
@Config(sdk = {36})
public class FeatureEmbedderPreprocessTest {

    private static final int INPUT = 224;
    private static final int AREA = INPUT * INPUT;

    // Must match FeatureEmbedder's ImageNet constants, and the MEAN/STD in the Python copies.
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private static Bitmap solid(int width, int height, int color) {
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];
        java.util.Arrays.fill(pixels, color);
        bmp.setPixels(pixels, 0, width, 0, 0, width, height);
        return bmp;
    }

    @Test
    public void producesAnNchwTensorOfExactlyOneNormalisedImage() {
        FloatBuffer out = FeatureEmbedder.preprocess(solid(INPUT, INPUT, Color.BLACK));
        assertEquals("tensor must be 3x224x224", 3 * AREA, out.remaining());
    }

    @Test
    public void aNonSquareImageStillYieldsTheSameTensorSize() {
        // The shorter side drives the resize and the longer one is centre-cropped away, so
        // the tensor size cannot depend on the input's aspect ratio.
        assertEquals(3 * AREA, FeatureEmbedder.preprocess(solid(640, 480, Color.WHITE)).remaining());
        assertEquals(3 * AREA, FeatureEmbedder.preprocess(solid(480, 640, Color.WHITE)).remaining());
    }

    @Test
    public void anImageSmallerThanTheInputIsUpscaledRatherThanPadded() {
        // max(INPUT, ...) in the resize exists for this: a crop smaller than 224 must still
        // fill the tensor, not leave a border of undefined pixels.
        assertEquals(3 * AREA, FeatureEmbedder.preprocess(solid(40, 40, Color.WHITE)).remaining());
    }

    @Test
    public void channelsArePlanarInRgbOrderAndImageNetNormalised() {
        // A colour with three distinct channels, so a swapped plane cannot pass unnoticed.
        int r = 255, g = 128, b = 0;
        FloatBuffer out = FeatureEmbedder.preprocess(solid(INPUT, INPUT, Color.rgb(r, g, b)));
        float[] values = new float[3 * AREA];
        out.get(values);

        float expectedR = ((r / 255f) - MEAN[0]) / STD[0];
        float expectedG = ((g / 255f) - MEAN[1]) / STD[1];
        float expectedB = ((b / 255f) - MEAN[2]) / STD[2];

        // Planar layout: the whole R plane, then G, then B — not interleaved per pixel.
        assertEquals(expectedR, values[0], 1e-4f);
        assertEquals(expectedR, values[AREA - 1], 1e-4f);
        assertEquals(expectedG, values[AREA], 1e-4f);
        assertEquals(expectedG, values[2 * AREA - 1], 1e-4f);
        assertEquals(expectedB, values[2 * AREA], 1e-4f);
        assertEquals(expectedB, values[3 * AREA - 1], 1e-4f);
    }

    @Test
    public void blackAndWhiteLandOnTheExpectedNormalisedExtremes() {
        // Guards the normalisation direction: dividing by STD instead of multiplying, or
        // dropping the mean subtraction, moves these and nothing else would notice.
        FloatBuffer black = FeatureEmbedder.preprocess(solid(INPUT, INPUT, Color.BLACK));
        float[] blackValues = new float[3 * AREA];
        black.get(blackValues);
        assertEquals(-MEAN[0] / STD[0], blackValues[0], 1e-4f);
        assertEquals(-MEAN[1] / STD[1], blackValues[AREA], 1e-4f);
        assertEquals(-MEAN[2] / STD[2], blackValues[2 * AREA], 1e-4f);

        FloatBuffer white = FeatureEmbedder.preprocess(solid(INPUT, INPUT, Color.WHITE));
        float[] whiteValues = new float[3 * AREA];
        white.get(whiteValues);
        assertEquals((1f - MEAN[0]) / STD[0], whiteValues[0], 1e-4f);
        assertEquals((1f - MEAN[1]) / STD[1], whiteValues[AREA], 1e-4f);
        assertEquals((1f - MEAN[2]) / STD[2], whiteValues[2 * AREA], 1e-4f);
    }

    @Test
    public void theBufferIsRewoundAndReadyToRead() {
        // It is handed straight to OnnxTensor.createTensor; a buffer left at its limit
        // would produce an empty tensor rather than an error.
        assertEquals(0, FeatureEmbedder.preprocess(solid(INPUT, INPUT, Color.WHITE)).position());
    }
}
