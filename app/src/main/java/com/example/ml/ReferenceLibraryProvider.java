package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.io.File;
import java.io.InputStream;

/**
 * Resolves which {@link ReferenceLibrary} to use for a given part, and builds the
 * bundled default library on-device (mirrors build_reference_library()).
 *
 * Resolution order for a part:
 *   1. Per-part library written by the in-app Reference Setup builder
 *      (filesDir/clip_lib/&lt;partId&gt;.json).
 *   2. The default library, cached at filesDir/clip_lib/_default.json.
 *      Built once, on first use, by embedding the bundled reference crops in
 *      assets/reference_crops/&lt;type&gt;/*.jpg with {@link ClipEmbedder}.
 *
 * Returns null when nothing is available (no model, no crops, nothing built);
 * callers then classify everything as "unknown".
 */
public final class ReferenceLibraryProvider {

    private static final String TAG = "RefLibProvider";
    private static final String DIR = "clip_lib";
    private static final String DEFAULT_FILE = "_default.json";
    private static final String CROPS_ASSET_DIR = "reference_crops";

    private ReferenceLibraryProvider() {}

    /** Directory holding per-part + default library JSON files. */
    public static File libDir(Context context) {
        File dir = new File(context.getFilesDir(), DIR);
        if (!dir.exists()) dir.mkdirs();
        return dir;
    }

    /** File path where the in-app builder saves a part's library. */
    public static File fileForPart(Context context, String partId) {
        return new File(libDir(context), safe(partId) + ".json");
    }

    /** Effective library for a part: per-part if present, else the default. */
    public static ReferenceLibrary effectiveFor(Context context, String partId) {
        ReferenceLibrary perPart = ReferenceLibrary.load(fileForPart(context, partId));
        if (perPart != null && !perPart.isEmpty()) return perPart;
        return getDefault(context);
    }

    /** Default library: cached JSON if present, otherwise bootstrap from bundled crops. */
    public static synchronized ReferenceLibrary getDefault(Context context) {
        File cache = new File(libDir(context), DEFAULT_FILE);
        ReferenceLibrary cached = ReferenceLibrary.load(cache);
        if (cached != null && !cached.isEmpty()) return cached;

        ReferenceLibrary built = bootstrapFromAssets(context);
        if (built != null && !built.isEmpty()) {
            built.save(cache);
            return built;
        }
        return null;
    }

    /** Embeds bundled reference crops (assets/reference_crops/&lt;type&gt;/*) into a library. */
    private static ReferenceLibrary bootstrapFromAssets(Context context) {
        ClipEmbedder embedder = ClipEmbedder.get(context);
        if (embedder == null) {
            Log.w(TAG, "No CLIP model available; cannot build default library.");
            return null;
        }
        ReferenceLibrary lib = new ReferenceLibrary();
        int total = 0;
        for (String type : ReferenceLibrary.FEATURE_TYPES) {
            String dir = CROPS_ASSET_DIR + "/" + type;
            String[] files;
            try {
                files = context.getAssets().list(dir);
            } catch (Exception e) {
                files = null;
            }
            if (files == null) continue;
            for (String name : files) {
                String lower = name.toLowerCase();
                if (!(lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png"))) {
                    continue;
                }
                try (InputStream is = context.getAssets().open(dir + "/" + name)) {
                    Bitmap bmp = BitmapFactory.decodeStream(is);
                    if (bmp == null) continue;
                    float[] emb = embedder.embed(bmp);
                    bmp.recycle();
                    if (emb != null) {
                        lib.addExample(type, emb);
                        total++;
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Failed to embed reference crop " + dir + "/" + name, e);
                }
            }
        }
        Log.d(TAG, "Default library bootstrap: " + total + " crops across "
                + lib.types().size() + " types.");
        return lib;
    }

    /** Clears the cached default library so it rebuilds on next use (e.g. after adding crops). */
    public static void invalidateDefault(Context context) {
        File cache = new File(libDir(context), DEFAULT_FILE);
        if (cache.exists()) //noinspection ResultOfMethodCallIgnored
            cache.delete();
    }

    private static String safe(String partId) {
        if (partId == null || partId.isEmpty()) return "unknown_part";
        return partId.replaceAll("[^A-Za-z0-9_.-]", "_");
    }
}
