package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.BitmapRegionDecoder;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.util.Log;

import com.example.ui.util.AssetBitmaps;

import java.io.Closeable;
import java.io.InputStream;

/**
 * Cuts feature crops out of the <em>source</em> photo rather than the downscaled copy the
 * detector ran on.
 *
 * <h3>Why</h3>
 * Detection runs at {@link HardwareFeatureExtractor#WORKING_MAX_DIM}, but the phone's photos
 * are around 4032x3024 — a 3.15x downscale before anything is cropped. A fastener that is 95
 * pixels wide in the original arrives as 30, and the encoder then has to upscale that ~7x to
 * its 224 input. Structure survives; fine texture does not, and fine texture is exactly what
 * separates a threaded screw from a smooth stud.
 *
 * Widening the crop instead would only trade the problem for a worse one: padding enough to
 * fill the frame pulls in neighbouring hardware, and framing that varies with how far away
 * the operator stood makes a prototype average over camera distance as much as over parts.
 * Reading the original pixels removes the trade rather than picking a side of it.
 *
 * The bundled reference crops are a median 236x187, so source-resolution crops also land in
 * the same detail range as the prototypes they are compared against.
 *
 * <h3>Memory</h3>
 * {@link BitmapRegionDecoder} decodes only the requested rectangle, so the full-size bitmap
 * — 48 MB as ARGB_8888 — is never materialised. Each region is additionally sampled down to
 * {@link #CROP_MAX_DIM}, which bounds a crop of a very large detection.
 *
 * <h3>Orientation</h3>
 * The working image has EXIF rotation applied; the region decoder does not, and reads raw
 * sensor coordinates. Boxes are therefore mapped back through that rotation before decoding
 * and the crop rotated forward afterwards. Getting this wrong would not degrade a crop, it
 * would cut a completely different part of the photo, so the mapping is a pure function with
 * its own tests.
 */
public final class SourceCropper implements Closeable {

    private static final String TAG = "SourceCropper";

    /**
     * Longest edge kept for a decoded crop. The encoder resizes to 224, so this is headroom
     * for its aspect-preserving resize rather than detail anyone sees — and it caps what a
     * detection covering half the frame can cost.
     */
    private static final int CROP_MAX_DIM = 512;

    private final BitmapRegionDecoder decoder;
    /** Raw source dimensions, before EXIF rotation. */
    private final int rawWidth;
    private final int rawHeight;
    /** EXIF rotation in degrees, one of 0/90/180/270. */
    private final int rotation;
    /** Upright source pixels per working pixel. Always >= 1 in practice, but not assumed. */
    private final float scale;

    private SourceCropper(BitmapRegionDecoder decoder, int rawWidth, int rawHeight,
                          int rotation, float scale) {
        this.decoder = decoder;
        this.rawWidth = rawWidth;
        this.rawHeight = rawHeight;
        this.rotation = rotation;
        this.scale = scale;
    }

    /**
     * Opens a cropper for {@code uriString}, or returns null when the source cannot be region
     * decoded or offers no more detail than the working image already has.
     *
     * Returning null is a normal outcome, not a failure: several bundled assets are smaller
     * than the working size, and the caller simply keeps cropping from the working image.
     */
    public static SourceCropper open(Context context, String uriString,
                                     int workingWidth, int workingHeight) {
        if (context == null || uriString == null || workingWidth <= 0 || workingHeight <= 0) {
            return null;
        }
        BitmapRegionDecoder decoder = null;
        try {
            int rotation = HardwareFeatureExtractor.exifRotation(context, uriString);
            try (InputStream is = HardwareFeatureExtractor.open(context, uriString)) {
                if (is == null) return null;
                decoder = newDecoder(is);
            }
            if (decoder == null) return null;

            int rawW = decoder.getWidth();
            int rawH = decoder.getHeight();
            if (rawW <= 0 || rawH <= 0) return null;

            // Upright dimensions are the raw ones with the axes swapped on a quarter turn.
            boolean quarterTurn = rotation == 90 || rotation == 270;
            int uprightW = quarterTurn ? rawH : rawW;
            float scale = (float) uprightW / workingWidth;
            if (scale <= 1.05f) {
                // Nothing meaningful to gain; upscaling from here would be a pure cost.
                decoder.recycle();
                return null;
            }
            Log.d(TAG, "Cropping from source " + rawW + "x" + rawH + " (rotation " + rotation
                    + "), " + String.format(java.util.Locale.US, "%.2fx", scale)
                    + " the working image");
            return new SourceCropper(decoder, rawW, rawH, rotation, scale);
        } catch (Throwable t) {
            // A content URI backed by a non-seekable stream, or a format the region decoder
            // does not handle, both land here. Neither is worth failing the inspection over.
            Log.d(TAG, "Source-resolution cropping unavailable for " + uriString + ": " + t);
            if (decoder != null) decoder.recycle();
            return null;
        }
    }

    @SuppressWarnings("deprecation")
    private static BitmapRegionDecoder newDecoder(InputStream is) throws Exception {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
            return BitmapRegionDecoder.newInstance(is);
        }
        return BitmapRegionDecoder.newInstance(is, false);
    }

    /**
     * Decodes the region of the source corresponding to a working-image box, with padding
     * already applied in working pixels. Returns null when the region cannot be read, which
     * the caller treats as "fall back to the working-image crop".
     */
    public Bitmap crop(Rect workingBox, int workingPadding) {
        if (workingBox == null) return null;
        try {
            int uprightW = Math.round(workingBox.width() * scale);
            int uprightH = Math.round(workingBox.height() * scale);
            int pad = Math.round(workingPadding * scale);
            Rect upright = new Rect(
                    Math.round(workingBox.left * scale) - pad,
                    Math.round(workingBox.top * scale) - pad,
                    Math.round(workingBox.left * scale) + uprightW + pad,
                    Math.round(workingBox.top * scale) + uprightH + pad);

            boolean quarterTurn = rotation == 90 || rotation == 270;
            int fullUprightW = quarterTurn ? rawHeight : rawWidth;
            int fullUprightH = quarterTurn ? rawWidth : rawHeight;
            clamp(upright, fullUprightW, fullUprightH);
            if (upright.width() < 8 || upright.height() < 8) return null;

            Rect raw = toRawRect(upright, rawWidth, rawHeight, rotation);
            clamp(raw, rawWidth, rawHeight);
            if (raw.width() < 8 || raw.height() < 8) return null;

            BitmapFactory.Options opts = new BitmapFactory.Options();
            opts.inSampleSize = AssetBitmaps.sampleSizeFor(raw.width(), raw.height(), CROP_MAX_DIM);
            Bitmap decoded = decoder.decodeRegion(raw, opts);
            if (decoded == null) return null;
            return rotate(decoded, rotation);
        } catch (Throwable t) {
            Log.d(TAG, "Region decode failed", t);
            return null;
        }
    }

    /**
     * Maps a rectangle in upright (EXIF-corrected) space to raw sensor space.
     *
     * Rotations are multiples of 90, so a rectangle maps to a rectangle exactly. Derived from
     * what {@link Matrix#postRotate} does to a bitmap: at 90 a source pixel (x, y) lands at
     * (rawHeight - 1 - y, x), and the mappings below are that relation inverted per quadrant.
     */
    static Rect toRawRect(Rect upright, int rawWidth, int rawHeight, int rotation) {
        switch (normalise(rotation)) {
            case 90:
                return new Rect(upright.top, rawHeight - upright.right,
                        upright.bottom, rawHeight - upright.left);
            case 180:
                return new Rect(rawWidth - upright.right, rawHeight - upright.bottom,
                        rawWidth - upright.left, rawHeight - upright.top);
            case 270:
                return new Rect(rawWidth - upright.bottom, upright.left,
                        rawWidth - upright.top, upright.right);
            default:
                return new Rect(upright);
        }
    }

    private static int normalise(int degrees) {
        int d = degrees % 360;
        if (d < 0) d += 360;
        return (d / 90) * 90;
    }

    private static void clamp(Rect r, int width, int height) {
        r.left = Math.max(0, Math.min(r.left, width));
        r.top = Math.max(0, Math.min(r.top, height));
        r.right = Math.max(r.left, Math.min(r.right, width));
        r.bottom = Math.max(r.top, Math.min(r.bottom, height));
    }

    private static Bitmap rotate(Bitmap src, int degrees) {
        int d = normalise(degrees);
        if (d == 0) return src;
        Matrix m = new Matrix();
        m.postRotate(d);
        Bitmap out = Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), m, true);
        if (out != src) src.recycle();
        return out;
    }

    @Override
    public void close() {
        try {
            decoder.recycle();
        } catch (Throwable ignored) {
            // Recycling a decoder that already failed is not worth reporting.
        }
    }
}
