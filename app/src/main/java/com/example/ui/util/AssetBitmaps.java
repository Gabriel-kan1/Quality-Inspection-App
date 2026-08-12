package com.example.ui.util;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.io.InputStream;

/**
 * Downsampled bitmap loading for asset images.
 *
 * The chassis assets are camera-resolution (several are 4032x3024), which decode to ~46MB
 * each as ARGB_8888. Loading a folder of them at full size for 48dp thumbnails costs
 * hundreds of megabytes — and because bitmap pixels live in native memory on Android 8+,
 * that never raises OutOfMemoryError. It just grows the process until the kernel's
 * low-memory killer terminates it, which looks like a frozen black screen.
 *
 * Always decode at the size actually being displayed.
 */
public final class AssetBitmaps {

    private static final String TAG = "AssetBitmaps";

    private AssetBitmaps() {}

    /**
     * Decodes an asset scaled down to roughly {@code reqPx} on its longest edge.
     *
     * Uses the standard two-pass approach: measure with inJustDecodeBounds, pick a
     * power-of-two inSampleSize, then decode for real. Returns null on failure.
     */
    public static Bitmap decodeSampled(Context context, String assetPath, int reqPx) {
        if (context == null || assetPath == null || reqPx <= 0) return null;

        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream is = context.getAssets().open(assetPath)) {
            BitmapFactory.decodeStream(is, null, bounds);
        } catch (Exception e) {
            Log.e(TAG, "Bounds pass failed for " + assetPath, e);
            return null;
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null;

        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inSampleSize = sampleSizeFor(bounds.outWidth, bounds.outHeight, reqPx);
        // Thumbnails don't need 4 bytes/px; RGB_565 halves the cost again and these are opaque JPEGs.
        opts.inPreferredConfig = Bitmap.Config.RGB_565;
        try (InputStream is = context.getAssets().open(assetPath)) {
            return BitmapFactory.decodeStream(is, null, opts);
        } catch (Exception e) {
            Log.e(TAG, "Decode failed for " + assetPath, e);
            return null;
        }
    }

    /**
     * Largest power-of-two sample size that keeps the longest edge >= reqPx.
     *
     * Public because every decode in the app needs this: BitmapFactory silently rounds
     * inSampleSize down to a power of two, so computing it any other way produces a
     * decode resolution that doesn't match what the caller asked for.
     */
    public static int sampleSizeFor(int width, int height, int reqPx) {
        int longest = Math.max(width, height);
        int sample = 1;
        while (longest / (sample * 2) >= reqPx) {
            sample *= 2;
        }
        return sample;
    }
}
