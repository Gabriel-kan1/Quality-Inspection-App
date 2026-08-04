package com.example.ml;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.zip.CRC32;

/**
 * Resolves which {@link ReferenceLibrary} to use for a given part, and builds the
 * bundled default library on-device (mirrors build_reference_library()).
 *
 * Resolution order for a part:
 *   1. Per-part library written by the in-app Reference Setup builder
 *      (filesDir/clip_lib/&lt;partId&gt;.json).
 *   2. The default library, cached at filesDir/clip_lib/_default.json.
 *      Built by embedding the bundled reference crops in
 *      assets/reference_crops/&lt;type&gt;/*.jpg with {@link FeatureEmbedder}.
 *
 * The default library is content-addressed: a hash of every crop's bytes under
 * assets/reference_crops/, plus {@link FeatureEmbedder#ENCODER_ID}, is stored alongside
 * the cache (_default.sig). Editing, adding, or removing crops -- or swapping the encoder
 * -- changes the hash, so the next call to {@link #getDefault} detects the mismatch and
 * rebuilds automatically. Mirrors the rebuild_needed check in build_reference_library() /
 * the Python script's REFERENCE_DIR watch, but content-aware rather than just
 * folder-name-aware.
 *
 * Returns null when nothing is available (no model, no crops, nothing built);
 * callers then classify everything as "unknown".
 */
public final class ReferenceLibraryProvider {

    private static final String TAG = "RefLibProvider";
    /**
     * Historical name — the first encoder was CLIP. Kept as-is deliberately: it is a path
     * on existing installs, and renaming it would orphan every per-part library inspectors
     * have already built rather than letting {@link ReferenceLibrary#fromJson} report the
     * encoder mismatch and prompt a rebuild.
     */
    private static final String DIR = "clip_lib";
    private static final String DEFAULT_FILE = "_default.json";
    private static final String DEFAULT_SIG_FILE = "_default.sig";
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

    /**
     * Default library: cached JSON if present AND the bundled crops haven't changed since
     * it was built, otherwise (re)bootstrap from assets/reference_crops/.
     */
    public static synchronized ReferenceLibrary getDefault(Context context) {
        String currentSig = computeCropsSignature(context);
        File cacheFile = new File(libDir(context), DEFAULT_FILE);
        File sigFile = new File(libDir(context), DEFAULT_SIG_FILE);

        ReferenceLibrary cached = ReferenceLibrary.load(cacheFile);
        String storedSig = readSignature(sigFile);
        if (cached != null && !cached.isEmpty() && currentSig.equals(storedSig)) {
            return cached;
        }

        if (cached != null && !currentSig.equals(storedSig)) {
            Log.d(TAG, "reference_crops/ changed since last build — rebuilding default library.");
        }

        ReferenceLibrary built = bootstrapFromAssets(context);
        if (built != null && !built.isEmpty()) {
            built.save(cacheFile);
            writeSignature(sigFile, currentSig);
            return built;
        }
        return null;
    }

    /** Embeds bundled reference crops (assets/reference_crops/&lt;type&gt;/*) into a library. */
    private static ReferenceLibrary bootstrapFromAssets(Context context) {
        FeatureEmbedder embedder = FeatureEmbedder.get(context);
        if (embedder == null) {
            Log.w(TAG, "No encoder model available; cannot build default library.");
            return null;
        }
        ReferenceLibrary lib = new ReferenceLibrary();
        int total = 0;
        for (String type : ReferenceLibrary.FEATURE_TYPES) {
            String dir = CROPS_ASSET_DIR + "/" + type;
            for (String name : listCropFiles(context, dir)) {
                try {
                    byte[] bytes = readAsset(context, dir + "/" + name);
                    Bitmap bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
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

    /**
     * Content hash of every crop under assets/reference_crops/, sorted for determinism,
     * plus the encoder identity.
     *
     * The encoder has to be in here: prototypes are only comparable to embeddings from
     * the model that produced them, so changing the model invalidates the cache just as
     * surely as editing a crop does. Hashing only the crops meant an encoder swap left a
     * stale cache that still matched its signature and was loaded without complaint.
     * {@link ReferenceLibrary#fromJson} independently rejects mismatched libraries; this
     * makes the rebuild happen for the right reason rather than as a side effect.
     */
    private static String computeCropsSignature(Context context) {
        CRC32 crc = new CRC32();
        crc.update((FeatureEmbedder.ENCODER_ID + ":").getBytes(StandardCharsets.UTF_8));
        for (String type : ReferenceLibrary.FEATURE_TYPES) {
            String dir = CROPS_ASSET_DIR + "/" + type;
            for (String name : listCropFiles(context, dir)) {
                try {
                    byte[] bytes = readAsset(context, dir + "/" + name);
                    crc.update((type + "/" + name + ":").getBytes(StandardCharsets.UTF_8));
                    crc.update(bytes);
                } catch (Exception e) {
                    Log.e(TAG, "Failed to hash reference crop " + dir + "/" + name, e);
                }
            }
        }
        return Long.toHexString(crc.getValue());
    }

    private static List<String> listCropFiles(Context context, String dir) {
        List<String> result = new ArrayList<>();
        String[] files;
        try {
            files = context.getAssets().list(dir);
        } catch (Exception e) {
            files = null;
        }
        if (files == null) return result;
        for (String name : files) {
            String lower = name.toLowerCase();
            if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png")) {
                result.add(name);
            }
        }
        Collections.sort(result);
        return result;
    }

    private static byte[] readAsset(Context context, String path) throws Exception {
        try (InputStream is = context.getAssets().open(path)) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream(Math.max(1024, is.available()));
            byte[] buf = new byte[8192];
            int n;
            while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
            return bos.toByteArray();
        }
    }

    private static String readSignature(File file) {
        if (!file.exists()) return null;
        try (InputStream is = new java.io.FileInputStream(file)) {
            return new String(readAllBytes(is), StandardCharsets.UTF_8).trim();
        } catch (Exception e) {
            return null;
        }
    }

    private static void writeSignature(File file, String signature) {
        try (FileOutputStream fos = new FileOutputStream(file)) {
            fos.write(signature.getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            Log.e(TAG, "Failed to write signature to " + file, e);
        }
    }

    private static byte[] readAllBytes(InputStream is) throws Exception {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int n;
        while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
        return bos.toByteArray();
    }

    /**
     * Forces a rebuild on next use, even if the crop signature hasn't changed. Not required
     * for normal crop edits (those are auto-detected by {@link #getDefault}); kept for manual
     * troubleshooting.
     */
    public static void invalidateDefault(Context context) {
        File cache = new File(libDir(context), DEFAULT_FILE);
        File sig = new File(libDir(context), DEFAULT_SIG_FILE);
        //noinspection ResultOfMethodCallIgnored
        if (cache.exists()) cache.delete();
        //noinspection ResultOfMethodCallIgnored
        if (sig.exists()) sig.delete();
    }

    private static String safe(String partId) {
        if (partId == null || partId.isEmpty()) return "unknown_part";
        return partId.replaceAll("[^A-Za-z0-9_.-]", "_");
    }
}
