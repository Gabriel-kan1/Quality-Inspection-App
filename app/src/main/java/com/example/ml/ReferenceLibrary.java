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

    // Starting thresholds (SIMILARITY_THRESHOLD in the Python CONFIG). These are only
    // the defaults — each library instance carries its own values so {@link ThresholdTuner}
    // can fit them per-part from that part's confirmed classification history.
    public static final float DEFAULT_SIMILARITY_THRESHOLD = 0.85f;
    public static final float DEFAULT_HIGH_CONFIDENCE = 0.92f;
    /**
     * Minimum gap between the best and second-best prototype score. Below this the
     * match is too close to call (e.g. stud vs bolt_head) and we report "unknown"
     * with the best guess as closestMatch, rather than silently picking a winner.
     */
    public static final float DEFAULT_MARGIN_THRESHOLD = 0.03f;

    private float similarityThreshold = DEFAULT_SIMILARITY_THRESHOLD;
    private float highConfidence = DEFAULT_HIGH_CONFIDENCE;
    private float marginThreshold = DEFAULT_MARGIN_THRESHOLD;

    /** Why a classification landed where it did — drives active-learning review selection. */
    public enum Reason {
        /** Accepted: cleared both the absolute threshold and the runner-up margin. */
        ACCEPTED,
        /** Best score was below the absolute similarity threshold. */
        BELOW_THRESHOLD,
        /** Score cleared the threshold but two types were too close to call. */
        AMBIGUOUS,
        /** No prototypes at all (no CLIP model, or nothing labelled yet). */
        NO_LIBRARY
    }

    public float getSimilarityThreshold() { return similarityThreshold; }
    public float getMarginThreshold() { return marginThreshold; }
    public float getHighConfidence() { return highConfidence; }

    public void setThresholds(float similarityThreshold, float marginThreshold) {
        this.similarityThreshold = similarityThreshold;
        this.marginThreshold = marginThreshold;
    }

    // Running accumulation so prototypes can be built incrementally (in-app builder
    // or offline bootstrap). prototype(type) = sums[type] / counts[type].
    private final Map<String, float[]> sums = new LinkedHashMap<>();
    private final Map<String, Integer> counts = new LinkedHashMap<>();

    public static final class Classification {
        public final String type;         // best type, or "unknown" if below threshold
        public final String closestMatch; // nearest type regardless of threshold
        public final float score;         // best cosine similarity
        public final String confidence;   // "high" | "medium" | "low"
        public final String runnerUp;     // second-best type, or null when only one prototype exists
        public final float runnerUpScore; // second-best cosine similarity, or -1
        public final Reason reason;
        /** Cosine similarity against every prototype, so the UI can show the full picture. */
        public final Map<String, Float> scores;

        Classification(String type, String closestMatch, float score, String confidence,
                       String runnerUp, float runnerUpScore, Reason reason,
                       Map<String, Float> scores) {
            this.type = type;
            this.closestMatch = closestMatch;
            this.score = score;
            this.confidence = confidence;
            this.runnerUp = runnerUp;
            this.runnerUpScore = runnerUpScore;
            this.reason = reason;
            this.scores = scores;
        }

        /** Gap to the runner-up; Float.MAX_VALUE when there is no runner-up to compete with. */
        public float margin() {
            return runnerUp == null ? Float.MAX_VALUE : score - runnerUpScore;
        }
    }

    public boolean isEmpty() {
        return counts.isEmpty();
    }

    public List<String> types() {
        return new ArrayList<>(counts.keySet());
    }

    /**
     * Accumulates one example embedding for a feature type into the running mean.
     * The embedding is L2-normalized first so every example contributes an equal
     * "vote" on direction regardless of its raw magnitude — otherwise a few
     * larger-magnitude crops can dominate the mean and drag the prototype toward
     * a neighboring class (this is what was causing stud crops to skew into the
     * bolt_head prototype despite having enough examples).
     */
    public void addExample(String type, float[] embedding) {
        if (type == null || embedding == null) return;
        float[] normalized = normalize(embedding);
        if (normalized == null) return;
        float[] sum = sums.get(type);
        if (sum == null) {
            sums.put(type, normalized);
            counts.put(type, 1);
        } else {
            int n = Math.min(sum.length, normalized.length);
            for (int i = 0; i < n; i++) sum[i] += normalized[i];
            counts.put(type, counts.get(type) + 1);
        }
    }

    /** Returns a unit-length copy of v, or null if v has zero magnitude. */
    private static float[] normalize(float[] v) {
        double normSq = 0;
        for (float x : v) normSq += (double) x * x;
        if (normSq == 0) return null;
        float norm = (float) Math.sqrt(normSq);
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) out[i] = v[i] / norm;
        return out;
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

    /**
     * Mirrors classify_crop(): cosine similarity vs each prototype + thresholding.
     * Also requires the winner to beat the runner-up by MARGIN_THRESHOLD — a high
     * absolute score isn't enough if a different type scored almost as high, since
     * that's exactly the case where two classes (e.g. stud vs bolt_head) sit close
     * together in embedding space and a small centroid shift picks the wrong one.
     */
    public Classification classify(float[] embedding) {
        if (embedding == null || counts.isEmpty()) {
            return new Classification(UNKNOWN, "none", 0f, "low", null, -1f, Reason.NO_LIBRARY,
                    new LinkedHashMap<>());
        }

        String bestType = null;
        float bestScore = -1f;
        String secondType = null;
        float secondScore = -1f;
        Map<String, Float> allScores = new LinkedHashMap<>();
        for (String type : counts.keySet()) {
            float score = cosine(embedding, prototype(type));
            allScores.put(type, score);
            if (score > bestScore) {
                secondType = bestType;
                secondScore = bestScore;
                bestScore = score;
                bestType = type;
            } else if (score > secondScore) {
                secondScore = score;
                secondType = type;
            }
        }

        String closest = bestType;
        String type = bestType;
        boolean belowThreshold = bestScore < similarityThreshold;
        // Inclusive: a gap sitting exactly on the threshold is a tie, not a pass. With a strict
        // "<" a 0.93-vs-0.92 pair cleared a 0.01 margin and was committed silently.
        boolean tooClose = secondType != null && (bestScore - secondScore) <= marginThreshold;

        Reason reason;
        if (belowThreshold) {
            reason = Reason.BELOW_THRESHOLD;
            type = UNKNOWN;
        } else if (tooClose) {
            reason = Reason.AMBIGUOUS;
            type = UNKNOWN;
        } else {
            reason = Reason.ACCEPTED;
        }

        String confidence;
        if (tooClose) confidence = "low";
        else if (bestScore > highConfidence) confidence = "high";
        else if (bestScore > similarityThreshold) confidence = "medium";
        else confidence = "low";

        return new Classification(type, closest, bestScore, confidence, secondType, secondScore,
                reason, allScores);
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
