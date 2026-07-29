package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.util.Locale;

/**
 * Writes classified crops to disk so the self-learning loop has an actual dataset,
 * not just moving-average prototypes.
 *
 * Crops live at filesDir/learning_crops/&lt;partId&gt;/&lt;timestamp&gt;_&lt;n&gt;.png and are
 * referenced by {@link com.example.data.entity.ClassificationEvent#getCropPath()}.
 * Keeping them on the filesystem (rather than as DB blobs) means they survive the
 * Room destructive-migration path, and gives a future off-device fine-tune something
 * to export directly.
 */
public final class LearningCropStore {

    private static final String TAG = "LearningCropStore";
    private static final String DIR = "learning_crops";
    /** Crops are only ever fed to a 224x224 CLIP encoder, so there's no point storing more. */
    private static final int MAX_EDGE = 256;

    private LearningCropStore() {}

    public static File dirFor(Context context, String partId) {
        File dir = new File(new File(context.getFilesDir(), DIR), safe(partId));
        if (!dir.exists()) dir.mkdirs();
        return dir;
    }

    /** Saves a crop and returns its absolute path, or null if it couldn't be written. */
    public static String save(Context context, String partId, Bitmap crop, int index) {
        if (crop == null || crop.isRecycled()) return null;
        try {
            File dir = dirFor(context, partId);
            String name = String.format(Locale.US, "%d_%02d.png", System.currentTimeMillis(), index);
            File out = new File(dir, name);
            Bitmap scaled = downscale(crop);
            try (FileOutputStream fos = new FileOutputStream(out)) {
                scaled.compress(Bitmap.CompressFormat.PNG, 100, fos);
            }
            if (scaled != crop) scaled.recycle();
            return out.getAbsolutePath();
        } catch (Exception e) {
            Log.e(TAG, "Failed to persist learning crop for part " + partId, e);
            return null;
        }
    }

    private static Bitmap downscale(Bitmap src) {
        int w = src.getWidth();
        int h = src.getHeight();
        int longest = Math.max(w, h);
        if (longest <= MAX_EDGE) return src;
        float scale = (float) MAX_EDGE / longest;
        return Bitmap.createScaledBitmap(src, Math.round(w * scale), Math.round(h * scale), true);
    }

    /** Total bytes currently held by the on-device learning dataset for a part. */
    public static long sizeOnDisk(Context context, String partId) {
        File dir = dirFor(context, partId);
        File[] files = dir.listFiles();
        if (files == null) return 0;
        long total = 0;
        for (File f : files) total += f.length();
        return total;
    }

    private static String safe(String partId) {
        if (partId == null || partId.isEmpty()) return "unknown_part";
        return partId.replaceAll("[^A-Za-z0-9_.-]", "_");
    }
}
