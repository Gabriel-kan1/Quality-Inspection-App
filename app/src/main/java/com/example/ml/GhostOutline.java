package com.example.ml;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import androidx.annotation.Nullable;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

/**
 * Turns a reference photograph into a transparent bitmap carrying only its edges.
 *
 * <p>This is what the inspection screen draws over the live camera image in
 * {@link com.example.ui.GhostMode#OUTLINE}. The whole point is what it leaves out: every
 * pixel that is not an edge is fully transparent, so the live image the operator is framing
 * is not dimmed, washed or tinted anywhere the reference has nothing to say. The edges
 * themselves are drawn in one solid colour, which makes them read as an annotation rather
 * than as a second photograph competing for the same pixels.
 *
 * <p>The map comes from {@link EdgeMethodLab#outlineEdges}, the Hybrid v6 stage, because it
 * is the cleanest map the project has: exact on every stud and screw photograph in the
 * ground-truth set. A speckled map would draw grime as if it were hardware to aim at.
 */
public final class GhostOutline {

    private static final String TAG = "GhostOutline";

    private GhostOutline() {}

    /**
     * Renders the outline, or returns null when it cannot.
     *
     * <p>Null rather than a blank bitmap so the caller can fall back to the photograph: an
     * empty overlay would look like "the reference has no edges", which is never true, rather
     * than "OpenCV is not available on this device", which is what actually happened.
     *
     * <p>Heavy — a full edge stage at working resolution. Call it off the main thread.
     *
     * @param reference the reference photograph, ideally at
     *                  {@link HardwareFeatureExtractor#WORKING_MAX_DIM} so the stage's
     *                  pixel-denominated kernels mean what they were tuned to mean
     * @param argb      the colour every edge pixel is painted; alpha is honoured
     */
    @Nullable
    public static Bitmap render(Bitmap reference, int argb) {
        if (!HardwareFeatureExtractor.isAvailable() || reference == null) return null;

        Mat src = new Mat();
        Mat edges = null;
        Mat rgba = null;
        try {
            Utils.bitmapToMat(reference, src);
            edges = EdgeMethodLab.outlineEdges(src);

            // bitmapToMat and matToBitmap both speak RGBA, so the scalar is R,G,B,A — not the
            // BGR OpenCV uses when it reads files itself.
            rgba = new Mat(edges.size(), CvType.CV_8UC4, new Scalar(0, 0, 0, 0));
            rgba.setTo(new Scalar(Color.red(argb), Color.green(argb), Color.blue(argb),
                    Color.alpha(argb)), edges);

            Bitmap out = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgba, out);
            return out;
        } catch (Throwable t) {
            // UnsatisfiedLinkError is an Error; catching Exception alone would let an ABI
            // mismatch kill the worker thread silently, as it once did elsewhere in this app.
            Log.e(TAG, "Could not render the ghost outline", t);
            return null;
        } finally {
            src.release();
            if (edges != null) edges.release();
            if (rgba != null) rgba.release();
        }
    }
}
