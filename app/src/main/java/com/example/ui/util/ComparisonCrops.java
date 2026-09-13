package com.example.ui.util;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.Log;

import com.example.ml.HardwareFeatureExtractor;
import com.example.ml.SourceCropper;

/**
 * Cuts the crop shown on one side of a size comparison, and says where the measured box sits
 * inside it.
 *
 * <h3>Why these are cut fresh rather than reused</h3>
 * The classifier's crops are already on disk, but they were framed for an encoder — 15% of
 * padding, and nothing recorded about where the box fell once that padding was clipped at a
 * photo edge. A comparison needs the opposite: enough surroundings for an operator to
 * recognise the fastener and its neighbours, and the box's position to the pixel, because the
 * box is the thing that was actually compared and it gets drawn on top. Cutting again from the
 * source, with the geometry carried along, costs one region decode per side.
 *
 * <h3>Source or working copy</h3>
 * The same rule as the classifier: the region decoder reads the original photo when it has
 * more detail than the working image, and the working copy is decoded and cropped otherwise
 * (several bundled assets are smaller than the working size). Either way {@link Crop#box} is
 * exact for the bitmap returned.
 */
public final class ComparisonCrops {

    private static final String TAG = "ComparisonCrops";

    /**
     * Padding around the box, as a fraction of its longer side. Deliberately far wider than
     * the classifier's 0.15: that figure keeps the encoder looking at hardware rather than
     * background, whereas here the background is the point — the neighbouring hole or the
     * panel edge is what tells an operator they are looking at the same place on the part.
     * A full box-width each side also keeps the panel filled for whichever crop is drawn
     * smaller: the pair is shown at a common scale, so the side with the smaller box shrinks
     * with the ratio and would otherwise sit in a field of black. The larger side simply
     * overflows its panel and is clipped.
     */
    public static final float PADDING_RATIO = 1.0f;

    /** Smallest padding in working pixels, so a tiny box still gets some context. */
    private static final int MIN_PADDING_PX = 8;

    /** One side of the comparison: the pixels, and the measured box within them. */
    public static final class Crop {
        public final Bitmap bitmap;
        /** The measured box inside {@link #bitmap}, in bitmap pixels. */
        public final RectF box;
        /** The measured box in working-image pixels — the size figure the operator is shown. */
        public final int boxWidthPx;
        public final int boxHeightPx;

        Crop(Bitmap bitmap, RectF box, int boxWidthPx, int boxHeightPx) {
            this.bitmap = bitmap;
            this.box = box;
            this.boxWidthPx = boxWidthPx;
            this.boxHeightPx = boxHeightPx;
        }

        public void recycle() {
            if (bitmap != null && !bitmap.isRecycled()) bitmap.recycle();
        }
    }

    private ComparisonCrops() {}

    /**
     * Cuts the crop for a fractional box of the photo at {@code uriString}.
     *
     * @param workingWidth  dimensions of the working image the fraction was measured in —
     *                      what turns it back into the pixels the region decoder scales from
     * @return the crop, or null when the photo cannot be read at all. Runs a decode, so call
     *         it off the main thread.
     */
    public static Crop cut(Context context, String uriString, int workingWidth, int workingHeight,
                           RectF fraction) {
        if (context == null || uriString == null || fraction == null
                || workingWidth <= 0 || workingHeight <= 0) {
            return null;
        }
        Rect box = workingBox(fraction, workingWidth, workingHeight);
        if (box.width() <= 0 || box.height() <= 0) return null;
        int padding = paddingFor(box);

        SourceCropper cropper = SourceCropper.open(context, uriString, workingWidth, workingHeight);
        if (cropper != null) {
            try {
                SourceCropper.Region region = cropper.cropRegion(box, padding);
                if (region != null) {
                    return new Crop(region.bitmap, region.box, box.width(), box.height());
                }
            } finally {
                cropper.close();
            }
        }
        return cutFromWorkingCopy(context, uriString, box, padding);
    }

    /** The fractional box in working pixels, exactly as the detector's own box would be. */
    static Rect workingBox(RectF fraction, int workingWidth, int workingHeight) {
        return new Rect(Math.round(fraction.left * workingWidth),
                Math.round(fraction.top * workingHeight),
                Math.round(fraction.right * workingWidth),
                Math.round(fraction.bottom * workingHeight));
    }

    static int paddingFor(Rect box) {
        return Math.max(MIN_PADDING_PX,
                Math.round(Math.max(box.width(), box.height()) * PADDING_RATIO));
    }

    /**
     * The padded box clipped to the photo — the region a working-copy crop covers. Pure, so
     * the box mapping below can be tested without decoding anything.
     */
    static Rect clippedRegion(Rect box, int padding, int width, int height) {
        return new Rect(Math.max(0, box.left - padding), Math.max(0, box.top - padding),
                Math.min(width, box.right + padding), Math.min(height, box.bottom + padding));
    }

    private static Crop cutFromWorkingCopy(Context context, String uriString, Rect box,
                                           int padding) {
        Bitmap working = HardwareFeatureExtractor.loadBitmap(context, uriString,
                HardwareFeatureExtractor.WORKING_MAX_DIM);
        if (working == null) return null;
        Bitmap crop = null;
        try {
            Rect region = clippedRegion(box, padding, working.getWidth(), working.getHeight());
            if (region.width() < 2 || region.height() < 2) return null;
            crop = Bitmap.createBitmap(working, region.left, region.top,
                    region.width(), region.height());
            RectF within = new RectF(box.left - region.left, box.top - region.top,
                    box.right - region.left, box.bottom - region.top);
            return new Crop(crop, within, box.width(), box.height());
        } catch (Throwable t) {
            Log.e(TAG, "Working-copy crop failed for " + uriString, t);
            return null;
        } finally {
            // createBitmap hands back the source itself when the region is the whole image,
            // in which case it is now the crop and must survive.
            if (crop != working) working.recycle();
        }
    }
}
