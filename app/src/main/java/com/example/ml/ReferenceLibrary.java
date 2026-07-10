package com.example.ml;

import android.graphics.Color;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Few-shot classifier library, mirroring build_reference_library() and
 * classify_crop() in python/EdgeComputingImprove.py.
 *
 * Holds one prototype embedding per feature type (the mean of that type's
 * example embeddings). Classification is cosine similarity of a query embedding
 * against each prototype, thresholded exactly as in the Python script.
 *
 * Note: text-prompt scoring is intentionally omitted — the Python config sets
 * TEXT_PROMPT_WEIGHT = 0.0, so combined score == image score.
 */
public final class ReferenceLibrary {

    private static final String TAG = "ReferenceLibrary";

    public static final String UNKNOWN = "unknown";

    /** Canonical feature types (Python REFERENCE_DIR subfolders / TEXT_PROMPTS keys). */
    public static final String[] FEATURE_TYPES =
            {"bolt_head", "hole", "rivet_head", "screw", "stud"};

    // Tunable thresholds (SIMILARITY_THRESHOLD in the Python CONFIG). Non-final so
    // they can be nudged after on-device calibration.
    public static float SIMILARITY_THRESHOLD = 0.75f;
    public static float HIGH_CONFIDENCE = 0.88f;

    // Running accumulation so prototypes can be built incrementally (in-app builder
    // or offline bootstrap). prototype(type) = sums[type] / counts[type].
    private final Map<String, float[]> sums = new LinkedHashMap<>();
    private final Map<String, Integer> counts = new LinkedHashMap<>();

    public static final class Classification {
        public final String type;         // best type, or "unknown" if below threshold
        public final String closestMatch; // nearest type regardless of threshold
        public final float score;         // best cosine similarity
        public final String confidence;   // "high" | "medium" | "low"

        Classification(String type, String closestMatch, float score, String confidence) {
            this.type = type;
            this.closestMatch = closestMatch;
            this.score = score;
            this.confidence = confidence;
        }
    }

    public boolean isEmpty() {
        return counts.isEmpty();
    }

    public List<String> types() {
        return new ArrayList<>(counts.keySet());
    }

    /** Accumulates one example embedding for a feature type into the running mean. */
    public void addExample(String type, float[] embedding) {
        if (type == null || embedding == null) return;
        float[] sum = sums.get(type);
        if (sum == null) {
            sums.put(type, embedding.clone());
            counts.put(type, 1);
        } else {
            int n = Math.min(sum.length, embedding.length);
            for (int i = 0; i < n; i++) sum[i] += embedding[i];
            counts.put(type, counts.get(type) + 1);
        }
    }

    /** Averaged prototype for a type, or null if none. */
    public float[] prototype(String type) {
        float[] sum = sums.get(type);
        Integer c = counts.get(type);
        if (sum == null || c == null || c == 0) return null;
        float[] proto = new float[sum.length];
        for (int i = 0; i < sum.length; i++) proto[i] = sum[i] / c;
        return proto;
    }

    /** Mirrors classify_crop(): cosine similarity vs each prototype + thresholding. */
    public Classification classify(float[] embedding) {
        if (embedding == null || counts.isEmpty()) {
            return new Classification(UNKNOWN, "none", 0f, "low");
        }

        String bestType = null;
        float bestScore = -1f;
        for (String type : counts.keySet()) {
            float score = cosine(embedding, prototype(type));
            if (score > bestScore) {
                bestScore = score;
                bestType = type;
            }
        }

        String closest = bestType;
        String type = bestType;
        if (bestScore < SIMILARITY_THRESHOLD) {
            type = UNKNOWN;
        }

        String confidence;
        if (bestScore > HIGH_CONFIDENCE) confidence = "high";
        else if (bestScore > SIMILARITY_THRESHOLD) confidence = "medium";
        else confidence = "low";

        return new Classification(type, closest, bestScore, confidence);
    }

    private static float cosine(float[] a, float[] b) {
        if (a == null || b == null) return -1f;
        int n = Math.min(a.length, b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0 || nb == 0) return -1f;
        return (float) (dot / (Math.sqrt(na) * Math.sqrt(nb)));
    }

    // ── Persistence ──────────────────────────────────────────────────────────
    // Stored as { "type": [floats...] } of averaged prototypes.

    public void save(File file) {
        try {
            File parent = file.getParentFile();
            if (parent != null && !parent.exists()) parent.mkdirs();
            JSONObject root = new JSONObject();
            for (String type : counts.keySet()) {
                float[] proto = prototype(type);
                JSONArray arr = new JSONArray();
                for (float v : proto) arr.put((double) v);
                root.put(type, arr);
            }
            try (FileOutputStream fos = new FileOutputStream(file)) {
                fos.write(root.toString().getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            Log.e(TAG, "save() failed for " + file, e);
        }
    }

    public static ReferenceLibrary load(File file) {
        if (file == null || !file.exists()) return null;
        try (InputStream is = new java.io.FileInputStream(file)) {
            return fromJson(readAll(is));
        } catch (Exception e) {
            Log.e(TAG, "load() failed for " + file, e);
            return null;
        }
    }

    public static ReferenceLibrary fromJson(String json) {
        try {
            JSONObject root = new JSONObject(json);
            ReferenceLibrary lib = new ReferenceLibrary();
            java.util.Iterator<String> keys = root.keys();
            while (keys.hasNext()) {
                String type = keys.next();
                JSONArray arr = root.getJSONArray(type);
                float[] proto = new float[arr.length()];
                for (int i = 0; i < arr.length(); i++) proto[i] = (float) arr.getDouble(i);
                // Load prototype as a single example (count = 1).
                lib.sums.put(type, proto);
                lib.counts.put(type, 1);
            }
            return lib.isEmpty() ? null : lib;
        } catch (Exception e) {
            Log.e(TAG, "fromJson() failed", e);
            return null;
        }
    }

    private static String readAll(InputStream is) throws Exception {
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
        return new String(bos.toByteArray(), StandardCharsets.UTF_8);
    }

    // ── Display helpers (used by the overlay + results screens) ───────────────

    private static final Map<String, Integer> COLORS = new LinkedHashMap<>();
    private static final Map<String, String> DISPLAY = new LinkedHashMap<>();
    static {
        COLORS.put("bolt_head", Color.rgb(56, 189, 248));   // sky blue
        COLORS.put("hole", Color.rgb(251, 146, 60));        // orange
        COLORS.put("rivet_head", Color.rgb(250, 204, 21));  // yellow
        COLORS.put("screw", Color.rgb(74, 222, 128));       // green
        COLORS.put("stud", Color.rgb(167, 139, 250));       // violet
        COLORS.put(UNKNOWN, Color.rgb(255, 0, 255));        // magenta (Python)

        DISPLAY.put("bolt_head", "Bolt Heads");
        DISPLAY.put("hole", "Holes");
        DISPLAY.put("rivet_head", "Rivet Heads");
        DISPLAY.put("screw", "Screws");
        DISPLAY.put("stud", "Studs");
        DISPLAY.put(UNKNOWN, "Unknown");
    }

    public static int colorFor(String type) {
        Integer c = COLORS.get(type);
        return c != null ? c : Color.rgb(128, 128, 128);
    }

    public static String displayName(String type) {
        String d = DISPLAY.get(type);
        return d != null ? d : type;
    }
}
